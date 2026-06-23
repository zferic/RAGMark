import torch

from ..timing import RAGTiming, reset_gpu_peak
from ..utils import extract_answer
from .base import Pipeline


class StandardPipeline(Pipeline):

    # -------------------------------------------------------------------------
    # SINGLE QUERY
    # -------------------------------------------------------------------------
    def run_single(self, question, tracer=None):
        timing = RAGTiming()
        timing.start("rag_total")
        # ----------------------------------------------------
        # 1. EMBEDDING (profiled)
        # ----------------------------------------------------
        if self.args.use_rag:
            reset_gpu_peak()
            timing.capture_gpu_memory("embed_gpu_before")
            if tracer: tracer.mark("embed_start")
            timing.start("embed")
            q_emb = self.embedder.embed(question)
            timing.end("embed")
            if tracer: tracer.mark("embed_end")
            timing.capture_gpu_memory("embed_gpu_after")

            # ----------------------------------------------------
            # 2. RETRIEVAL (FAISS)
            # ----------------------------------------------------
            if self.retriever.is_gpu_index:
                if not q_emb.is_cuda:
                    q_emb = q_emb.to(self.index_device)
            else:
                if q_emb.is_cuda:
                    q_emb = q_emb.cpu()

            timing.capture_all_nvml_memory("faiss_nvml_before")
            if tracer: tracer.mark("faiss_start")
            timing.start("faiss")
            docs = self.retriever.retrieve(q_emb, k=self.args.rag_top_k)
            timing.end("faiss")
            if tracer: tracer.mark("faiss_end")
            timing.capture_all_nvml_memory("faiss_nvml_after")

            # ----------------------------------------------------
            # 3. RERANKING (optional)
            # ----------------------------------------------------
            if self.reranker:
                if tracer: tracer.mark("rerank_start")
                timing.start("rerank")
                docs = self.reranker.rerank(question, docs, top_n=self.args.rerank_top_n)
                timing.end("rerank")
                if tracer: tracer.mark("rerank_end")

            context = "\n".join(docs)
            timing.record_tokens("context_tokens_before", len(self.generator.tokenizer.tokenize(context)))

            # ----------------------------------------------------
            # 3. COMPRESSION (optional)
            # ----------------------------------------------------
            if self.compressor:
                if tracer: tracer.mark("compress_start")
                timing.start("compress")
                context = self.compressor.compress(context)
                timing.end("compress")
                if tracer: tracer.mark("compress_end")

            timing.record_tokens("context_tokens_after", len(self.generator.tokenizer.tokenize(context)))
        else:
            docs = ""
            timing.capture_all_nvml_memory("faiss_nvml_after")

        # ----------------------------------------------------
        # 4. GENERATION (profiled)
        # ----------------------------------------------------
        reset_gpu_peak(self.gen_device)
        timing.capture_gpu_memory("gen_gpu_before", self.gen_device)
        timing.capture_all_gpu_memory("gen_gpu_all_before")

        if tracer: tracer.mark("generate_start")
        gen_result = self.generator.generate_with_ttft(
            question,
            context if self.args.use_rag else None
        )
        if tracer: tracer.mark("generate_end")
        timing.capture_all_gpu_memory("gen_gpu_all_after")
        timing.capture_gpu_memory("gen_gpu_after")
        timing.end("rag_total")

        raw_output = gen_result["response"]
        answer = extract_answer(raw_output)

        print("Answer:", answer)

        timing.record("ttft", gen_result["ttft"])
        timing.record("decode_time", gen_result["decode_time"])
        timing.record("generate_total_time", gen_result["total_time"])
        timing.record("generate", gen_result["total_time"])
        timing.record("tokens_generated", gen_result["tokens_generated"])
        timing.record("tokens_per_second", gen_result["tokens_per_second"])

        # Aliases with _iter_0 suffix for compatibility with iterative pipeline timing schema
        for key in list(timing.data.keys()):
            if not key.endswith("_iter_0"):
                timing.data[f"{key}_iter_0"] = timing.data[key]

        return answer, docs, raw_output, timing

    # -------------------------------------------------------------------------
    # BATCH
    # -------------------------------------------------------------------------
    def run_batch(self, questions, tracer=None):
        timing = RAGTiming()
        timing.start("rag_total")

        # -------------------------------------------------------------------------
        # 1. BATCH EMBEDDING
        # -------------------------------------------------------------------------
        if self.args.use_rag:
            timing.reset_gpu_peak(self.embed_device)
            timing.capture_gpu_memory("embed_gpu_before", self.embed_device)

            if tracer: tracer.mark("embed_start")
            timing.start("embed")
            q_embs = self.embedder.batch_embed(questions)
            timing.end("embed")
            if tracer: tracer.mark("embed_end")

            timing.capture_gpu_memory("embed_gpu_after", self.embed_device)

            # Ensure embeddings are on correct device for FAISS
            if self.retriever.is_gpu_index:
                if not q_embs.is_cuda:
                    q_embs = q_embs.to(self.index_device)
            else:
                if q_embs.is_cuda:
                    q_embs = q_embs.cpu()

            # -------------------------------------------------------------------------
            # 2. BATCH RETRIEVAL (FAISS)
            # -------------------------------------------------------------------------

            if self.retriever.is_gpu_index:
                timing.reset_gpu_peak(self.retriever.device)
                timing.capture_gpu_memory("faiss_gpu_before", self.retriever.device)

            timing.capture_all_nvml_memory("faiss_nvml_before")
            if tracer: tracer.mark("faiss_start")
            timing.start("faiss")
            docs_batch = self.retriever.batch_retrieve(q_embs, k=self.args.rag_top_k)
            timing.end("faiss")
            if tracer: tracer.mark("faiss_end")
            timing.capture_all_nvml_memory("faiss_nvml_after")

            if self.retriever.is_gpu_index:
                timing.capture_gpu_memory("faiss_gpu_after", self.retriever.device)

            # -------------------------------------------------------------------------
            # 3. RERANKING (optional)
            # -------------------------------------------------------------------------
            if self.reranker:
                if tracer: tracer.mark("rerank_start")
                timing.start("rerank")
                docs_batch = self.reranker.rerank_batch(questions, docs_batch, top_n=self.args.rerank_top_n)
                timing.end("rerank")
                if tracer: tracer.mark("rerank_end")

            # -------------------------------------------------------------------------
            # 4. PREPARE CONTEXT (with compression)
            # -------------------------------------------------------------------------
            contexts = []
            before_tokens = 0
            after_tokens = 0

            for i, docs in enumerate(docs_batch):
                ctx = "\n".join(docs)
                before_tokens += len(self.generator.tokenizer.tokenize(ctx))

                if self.compressor:
                    timing.start(f"compress_{i}")
                    ctx = self.compressor.compress(ctx, question=questions[i])
                    timing.end(f"compress_{i}")

                after_tokens += len(self.generator.tokenizer.tokenize(ctx))
                contexts.append(ctx)

            timing.record_tokens("context_tokens_before", before_tokens)
            timing.record_tokens("context_tokens_after", after_tokens)

        # -------------------------------------------------------------------------
        # 4. TRUE BATCH GENERATION
        # -------------------------------------------------------------------------

        # Build prompts
        if self.args.use_rag:
            prompts = [
                self.generator.build_prompt(q, ctx)
                for q, ctx in zip(questions, contexts)
            ]
        else:
            prompts = [self.generator.build_prompt(q, None) for q in questions]
            docs_batch = []

        # Debug prints (optional)
        if len(prompts) >= 3:
            print("Prompts[0]:", prompts[0])
            print("Prompts[1]:", prompts[1])
            print("Prompts[2]:", prompts[2])

        # Tokenize as a batch (CPU → generator GPU)
        timing.start("tokenize")

        inputs = self.generator.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if self.gen_device is not None:
            inputs = inputs.to(self.gen_device)

        timing.end("tokenize")

        # -------------------------------------------------------------------------
        # Generation (GPU)
        # -------------------------------------------------------------------------
        if self.gen_device is not None:
            timing.reset_gpu_peak(self.gen_device)
            timing.capture_gpu_memory("gen_gpu_before", self.gen_device)
        timing.capture_all_gpu_memory("gen_gpu_all_before")

        timing.record("input_tokens_total", inputs["input_ids"].shape[0] * inputs["input_ids"].shape[1])

        if tracer: tracer.mark("generate_start")
        timing.start("generate")
        with torch.no_grad():
            outputs = self.generator.model.generate(
                **inputs,
                max_new_tokens=32,
                pad_token_id=self.generator.tokenizer.pad_token_id,
                eos_token_id=self.generator.tokenizer.eos_token_id,
            )
        timing.end("generate")
        if tracer: tracer.mark("generate_end")

        if self.gen_device is not None:
            timing.capture_gpu_memory("gen_gpu_after", self.gen_device)
        timing.capture_all_gpu_memory("gen_gpu_all_after")

        prompt_len = inputs["input_ids"].shape[-1]
        tokens_generated = sum(o.shape[-1] - prompt_len for o in outputs)
        timing.record("tokens_generated", tokens_generated)
        timing.record("tokens_per_second", tokens_generated / timing.data["generate"] if timing.data["generate"] > 0 else 0)

        # -------------------------------------------------------------------------
        # Decode (CPU)
        # -------------------------------------------------------------------------
        timing.start("decode")
        answers = [
            extract_answer(
                self.generator.tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            )
            for o in outputs
        ]
        timing.end("decode")

        # -------------------------------------------------------------------------
        # Total time
        # -------------------------------------------------------------------------
        timing.end("rag_total")

        # Debug prints (optional)
        if len(answers) >= 3:
            print("Answers[0]:", answers[0])
            print("Answers[1]:", answers[1])
            print("Answers[2]:", answers[2])

        # Aliases with _iter_0 suffix for compatibility with iterative pipeline timing schema
        for key in list(timing.data.keys()):
            if not key.endswith("_iter_0"):
                timing.data[f"{key}_iter_0"] = timing.data[key]

        return answers, docs_batch, timing
