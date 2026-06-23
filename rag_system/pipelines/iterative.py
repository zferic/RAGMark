import torch

from ..timing import RAGTiming, reset_gpu_peak
from ..utils import extract_answer
from .base import Pipeline


class IterativePipeline(Pipeline):
    """
    Iterative retrieval pipeline.

    Each round:
      - Round 0: query = original question
      - Round i>0: query = "{question} {previous answer}"
      - embed query → retrieve → (compress) → generate intermediate answer

    Final answer is the last round's generation.
    Timing keys are suffixed with _iter_{i} per round, plus top-level keys
    matching the standard pipeline format for CSV compatibility.
    """

    # -------------------------------------------------------------------------
    # SINGLE QUERY
    # -------------------------------------------------------------------------
    def run_single(self, question, tracer=None):
        timing = RAGTiming()
        timing.start("rag_total")

        iter_num = getattr(self.args, "iter_num", 2)

        query = question
        docs = []
        context = ""
        gen_result = None

        for iter_idx in range(iter_num):

            # ------------------------------------------------------------------
            # 1. EMBEDDING
            # ------------------------------------------------------------------
            reset_gpu_peak()
            timing.capture_gpu_memory(f"embed_gpu_before_iter_{iter_idx}")
            if tracer: tracer.mark(f"embed_start_iter_{iter_idx}")
            timing.start(f"embed_iter_{iter_idx}")
            q_emb = self.embedder.embed(query)
            timing.end(f"embed_iter_{iter_idx}")
            if tracer: tracer.mark(f"embed_end_iter_{iter_idx}")
            timing.capture_gpu_memory(f"embed_gpu_after_iter_{iter_idx}")

            # ------------------------------------------------------------------
            # 2. RETRIEVAL (FAISS)
            # ------------------------------------------------------------------
            if self.retriever.is_gpu_index:
                if not q_emb.is_cuda:
                    q_emb = q_emb.to(self.index_device)
            else:
                if q_emb.is_cuda:
                    q_emb = q_emb.cpu()

            timing.capture_all_nvml_memory(f"faiss_nvml_before_iter_{iter_idx}")
            if tracer: tracer.mark(f"faiss_start_iter_{iter_idx}")
            timing.start(f"faiss_iter_{iter_idx}")
            docs = self.retriever.retrieve(q_emb, k=self.args.rag_top_k)
            timing.end(f"faiss_iter_{iter_idx}")
            if tracer: tracer.mark(f"faiss_end_iter_{iter_idx}")
            timing.capture_all_nvml_memory(f"faiss_nvml_after_iter_{iter_idx}")

            # ------------------------------------------------------------------
            # 3. RERANKING (optional)
            # ------------------------------------------------------------------
            if self.reranker:
                if tracer: tracer.mark(f"rerank_start_iter_{iter_idx}")
                timing.start(f"rerank_iter_{iter_idx}")
                docs = self.reranker.rerank(question, docs, top_n=self.args.rerank_top_n)
                timing.end(f"rerank_iter_{iter_idx}")
                if tracer: tracer.mark(f"rerank_end_iter_{iter_idx}")

            context = "\n".join(docs)
            timing.record_tokens(f"context_tokens_before_iter_{iter_idx}",
                                 len(self.generator.tokenizer.tokenize(context)))

            # ------------------------------------------------------------------
            # 3. COMPRESSION
            # ------------------------------------------------------------------
            if self.compressor:
                if tracer: tracer.mark(f"compress_start_iter_{iter_idx}")
                timing.start(f"compress_iter_{iter_idx}")
                context = self.compressor.compress(context)
                timing.end(f"compress_iter_{iter_idx}")
                if tracer: tracer.mark(f"compress_end_iter_{iter_idx}")

            timing.record_tokens(f"context_tokens_after_iter_{iter_idx}",
                                 len(self.generator.tokenizer.tokenize(context)))

            # ------------------------------------------------------------------
            # 4. GENERATION
            # ------------------------------------------------------------------
            reset_gpu_peak(self.gen_device)
            timing.capture_gpu_memory(f"gen_gpu_before_iter_{iter_idx}", self.gen_device)
            timing.capture_all_gpu_memory(f"gen_gpu_all_before_iter_{iter_idx}")

            is_last = (iter_idx == iter_num - 1)
            prompt_type = "iterative_final" if is_last else "iterative_reasoning"
            if tracer: tracer.mark(f"generate_start_iter_{iter_idx}")
            gen_result = self.generator.generate_with_ttft(question, context, prompt_type=prompt_type)
            if tracer: tracer.mark(f"generate_end_iter_{iter_idx}")

            timing.capture_all_gpu_memory(f"gen_gpu_all_after_iter_{iter_idx}")
            timing.capture_gpu_memory(f"gen_gpu_after_iter_{iter_idx}", self.gen_device)

            intermediate = extract_answer(gen_result["response"])

            timing.record(f"ttft_iter_{iter_idx}", gen_result["ttft"])
            timing.record(f"decode_time_iter_{iter_idx}", gen_result["decode_time"])
            timing.record(f"generate_total_time_iter_{iter_idx}", gen_result["total_time"])
            timing.record(f"tokens_generated_iter_{iter_idx}", gen_result["tokens_generated"])
            timing.record(f"tokens_per_second_iter_{iter_idx}", gen_result["tokens_per_second"])

            top_doc = docs[0][:120] + "..." if docs and len(docs[0]) > 120 else (docs[0] if docs else "")
            print(f"\n--- iter {iter_idx} ---")
            print(f"  query : {query}")
            print(f"  top doc: {top_doc}")
            print(f"  answer : {intermediate}")

            # update query for next iteration
            if iter_idx < iter_num - 1:
                query = f"{question} {intermediate}"

        timing.end("rag_total")

        raw_output = gen_result["response"]
        answer = extract_answer(raw_output)

        # top-level keys matching standard pipeline format for CSV compatibility
        timing.record("ttft", gen_result["ttft"])
        timing.record("decode_time", gen_result["decode_time"])
        timing.record("generate_total_time", gen_result["total_time"])
        timing.record("tokens_generated", gen_result["tokens_generated"])
        timing.record("tokens_per_second", gen_result["tokens_per_second"])
        timing.record_tokens("context_tokens_before", len(self.generator.tokenizer.tokenize("\n".join(docs))))
        timing.record_tokens("context_tokens_after", len(self.generator.tokenizer.tokenize(context)))

        return answer, docs, raw_output, timing

    # -------------------------------------------------------------------------
    # BATCH
    # -------------------------------------------------------------------------
    def run_batch(self, questions, tracer=None):
        timing = RAGTiming()
        timing.start("rag_total")

        iter_num = getattr(self.args, "iter_num", 2)

        queries = list(questions)
        docs_batch = []
        contexts = []
        past_answers = [""] * len(questions)

        for iter_idx in range(iter_num):

            # ------------------------------------------------------------------
            # 1. BATCH EMBEDDING
            # ------------------------------------------------------------------
            timing.reset_gpu_peak(self.embed_device)
            timing.capture_gpu_memory(f"embed_gpu_before_iter_{iter_idx}", self.embed_device)
            if tracer: tracer.mark(f"embed_start_iter_{iter_idx}")
            timing.start(f"embed_iter_{iter_idx}")
            q_embs = self.embedder.batch_embed(queries)
            timing.end(f"embed_iter_{iter_idx}")
            if tracer: tracer.mark(f"embed_end_iter_{iter_idx}")
            timing.capture_gpu_memory(f"embed_gpu_after_iter_{iter_idx}", self.embed_device)

            # ------------------------------------------------------------------
            # 2. BATCH RETRIEVAL (FAISS)
            # ------------------------------------------------------------------
            if self.retriever.is_gpu_index:
                if not q_embs.is_cuda:
                    q_embs = q_embs.to(self.index_device)
            else:
                if q_embs.is_cuda:
                    q_embs = q_embs.cpu()

            if self.retriever.is_gpu_index:
                timing.reset_gpu_peak(self.retriever.device)
                timing.capture_gpu_memory(f"faiss_gpu_before_iter_{iter_idx}", self.retriever.device)

            timing.capture_all_nvml_memory(f"faiss_nvml_before_iter_{iter_idx}")
            if tracer: tracer.mark(f"faiss_start_iter_{iter_idx}")
            timing.start(f"faiss_iter_{iter_idx}")
            docs_batch = self.retriever.batch_retrieve(q_embs, k=self.args.rag_top_k)
            timing.end(f"faiss_iter_{iter_idx}")
            if tracer: tracer.mark(f"faiss_end_iter_{iter_idx}")
            timing.capture_all_nvml_memory(f"faiss_nvml_after_iter_{iter_idx}")

            if self.retriever.is_gpu_index:
                timing.capture_gpu_memory(f"faiss_gpu_after_iter_{iter_idx}", self.retriever.device)

            # ------------------------------------------------------------------
            # 3. RERANKING (optional)
            # ------------------------------------------------------------------
            if self.reranker:
                if tracer: tracer.mark(f"rerank_start_iter_{iter_idx}")
                timing.start(f"rerank_iter_{iter_idx}")
                docs_batch = self.reranker.rerank_batch(list(questions), docs_batch, top_n=self.args.rerank_top_n)
                timing.end(f"rerank_iter_{iter_idx}")
                if tracer: tracer.mark(f"rerank_end_iter_{iter_idx}")

            # ------------------------------------------------------------------
            # 4. PREPARE CONTEXT (with compression)
            # ------------------------------------------------------------------
            contexts = []
            before_tokens = 0
            after_tokens = 0

            for i, docs in enumerate(docs_batch):
                ctx = "\n".join(docs)
                before_tokens += len(self.generator.tokenizer.tokenize(ctx))

                if self.compressor:
                    timing.start(f"compress_{i}_iter_{iter_idx}")
                    ctx = self.compressor.compress(ctx, question=questions[i])
                    timing.end(f"compress_{i}_iter_{iter_idx}")

                after_tokens += len(self.generator.tokenizer.tokenize(ctx))
                contexts.append(ctx)

            timing.record_tokens(f"context_tokens_before_iter_{iter_idx}", before_tokens)
            timing.record_tokens(f"context_tokens_after_iter_{iter_idx}", after_tokens)

            # ------------------------------------------------------------------
            # 4. BATCH GENERATION
            # ------------------------------------------------------------------
            prompts = [
                self.generator.build_prompt(q, ctx)
                for q, ctx in zip(questions, contexts)
            ]

            timing.start(f"tokenize_iter_{iter_idx}")
            inputs = self.generator.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if self.gen_device is not None:
                inputs = inputs.to(self.gen_device)
            timing.end(f"tokenize_iter_{iter_idx}")

            if self.gen_device is not None:
                timing.reset_gpu_peak(self.gen_device)
                timing.capture_gpu_memory(f"gen_gpu_before_iter_{iter_idx}", self.gen_device)

            if tracer: tracer.mark(f"generate_start_iter_{iter_idx}")
            timing.start(f"generate_iter_{iter_idx}")
            with torch.no_grad():
                outputs = self.generator.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    pad_token_id=self.generator.tokenizer.pad_token_id,
                    eos_token_id=self.generator.tokenizer.eos_token_id,
                )
            timing.end(f"generate_iter_{iter_idx}")
            if tracer: tracer.mark(f"generate_end_iter_{iter_idx}")

            if self.gen_device is not None:
                timing.capture_gpu_memory(f"gen_gpu_after_iter_{iter_idx}", self.gen_device)

            prompt_len = inputs["input_ids"].shape[-1]
            past_answers = [
                extract_answer(self.generator.tokenizer.decode(o[prompt_len:], skip_special_tokens=True))
                for o in outputs
            ]

            print(f"[iter {iter_idx}] {len(past_answers)} answers generated")

            # update queries for next iteration
            if iter_idx < iter_num - 1:
                queries = [f"{q} {a}" for q, a in zip(questions, past_answers)]

        timing.end("rag_total")

        # top-level keys matching standard pipeline format for CSV compatibility
        timing.record_tokens("context_tokens_before", before_tokens)
        timing.record_tokens("context_tokens_after", after_tokens)

        return past_answers, docs_batch, timing
