"""Built-in node types: embed, retrieve, rerank, context, compress, generate."""
import torch

from ..generation import TTFTRecorder, itl_stats
from ..timing import reset_gpu_peak
from ..utils import extract_answer
from .nodes import Node, register_node, register_condition


def _ntok(generator, text):
    return len(generator.tokenizer.tokenize(text))


@register_node("embed")
class EmbedNode(Node):
    component = "embedder"

    def run(self, state, ctx):
        t, sfx, e = ctx.timing, ctx.sfx, ctx.comp(self)
        if ctx.batch:
            dev = ctx.devices["embed"]
            t.reset_gpu_peak(dev)
            t.capture_gpu_memory(f"embed_gpu_before{sfx}", dev)
            with ctx.stage("embed"):
                state["embs"] = e.batch_embed(state["queries"])
            t.capture_gpu_memory(f"embed_gpu_after{sfx}", dev)
        else:
            reset_gpu_peak()
            t.capture_gpu_memory(f"embed_gpu_before{sfx}")
            with ctx.stage("embed"):
                state["embs"] = e.embed(state["queries"][0])
            t.capture_gpu_memory(f"embed_gpu_after{sfx}")


@register_node("retrieve")
class RetrieveNode(Node):
    """FAISS lookup. params: k (default: args.rag_top_k)."""
    component = "retriever"

    def run(self, state, ctx):
        t, sfx, r = ctx.timing, ctx.sfx, ctx.comp(self)
        k = int(self.params.get("k", ctx.args.rag_top_k))
        embs = state["embs"]
        if r.is_gpu_index:
            if not embs.is_cuda:
                embs = embs.to(ctx.devices["index"])
        elif embs.is_cuda:
            embs = embs.cpu()

        if ctx.batch and r.is_gpu_index:
            t.reset_gpu_peak(r.device)
            t.capture_gpu_memory(f"faiss_gpu_before{sfx}", r.device)
        t.capture_all_nvml_memory(f"faiss_nvml_before{sfx}")
        with ctx.stage("faiss"):
            if ctx.batch:
                state["docs"] = r.batch_retrieve(embs, k=k)
            else:
                state["docs"] = r.retrieve(embs, k=k)
        t.capture_all_nvml_memory(f"faiss_nvml_after{sfx}")
        if ctx.batch and r.is_gpu_index:
            t.capture_gpu_memory(f"faiss_gpu_after{sfx}", r.device)


@register_node("rerank")
class RerankNode(Node):
    """Cross-encoder rerank against the ORIGINAL question. params: top_n."""
    component = "reranker"

    def run(self, state, ctx):
        rr = ctx.comp(self)
        top_n = int(self.params.get("top_n", ctx.args.rerank_top_n))
        with ctx.stage("rerank"):
            if ctx.batch:
                state["docs"] = rr.rerank_batch(list(state["questions"]), state["docs"], top_n=top_n)
            else:
                state["docs"] = rr.rerank(state["questions"][0], state["docs"], top_n=top_n)


@register_node("context")
class ContextNode(Node):
    """Join retrieved docs into a context string and record pre-compression token counts."""
    component = "generator"  # only used for its tokenizer

    def run(self, state, ctx):
        g = ctx.comp(self)
        if ctx.batch:
            state["contexts"] = ["\n".join(d) for d in state["docs"]]
            before = sum(_ntok(g, c) for c in state["contexts"])
        else:
            state["contexts"] = ["\n".join(state["docs"])]
            before = _ntok(g, state["contexts"][0])
        ctx.timing.record_tokens(f"context_tokens_before{ctx.sfx}", before)


@register_node("compress")
class CompressNode(Node):
    component = "compressor"

    def run(self, state, ctx):
        c, t, sfx = ctx.comp(self), ctx.timing, ctx.sfx
        if ctx.batch:
            for i, text in enumerate(state["contexts"]):
                t.start(f"compress_{i}{sfx}")
                state["contexts"][i] = c.compress(text, question=state["questions"][i])
                t.end(f"compress_{i}{sfx}")
        else:
            with ctx.stage("compress"):
                state["contexts"][0] = c.compress(state["contexts"][0])


@register_node("generate")
class GenerateNode(Node):
    """
    LLM generation. params: prompt_type = standard | iterative_reasoning |
    iterative_final | auto (reasoning on early rounds, final on the last).
    Batch mode always uses the standard prompt.
    """
    component = "generator"

    def run(self, state, ctx):
        g, t, sfx = ctx.comp(self), ctx.timing, ctx.sfx
        contexts = state.get("contexts")
        if contexts is not None:
            t.record_tokens(f"context_tokens_after{sfx}",
                            sum(_ntok(g, c) for c in contexts))
        if ctx.batch:
            self._run_batch(state, ctx, g, contexts)
        else:
            self._run_single(state, ctx, g, contexts)

    def _run_single(self, state, ctx, g, contexts):
        t, sfx, dev = ctx.timing, ctx.sfx, ctx.devices["gen"]
        if contexts is None:
            # no retrieval in this graph; keep the NVML baseline column populated
            t.capture_all_nvml_memory(f"faiss_nvml_after{sfx}")
        prompt_type = self.params.get("prompt_type", "standard")
        if prompt_type == "auto":
            prompt_type = "iterative_final" if ctx.last_iter else "iterative_reasoning"

        reset_gpu_peak(dev)
        t.capture_gpu_memory(f"gen_gpu_before{sfx}", dev)
        t.capture_all_gpu_memory(f"gen_gpu_all_before{sfx}")
        ctx.mark("generate_start")
        res = g.generate_with_ttft(state["questions"][0],
                                   contexts[0] if contexts is not None else None,
                                   prompt_type=prompt_type)
        ctx.mark("generate_end")
        t.capture_all_gpu_memory(f"gen_gpu_all_after{sfx}")
        t.capture_gpu_memory(f"gen_gpu_after{sfx}", dev)

        t.record(f"ttft{sfx}", res["ttft"])
        t.record(f"decode_time{sfx}", res["decode_time"])
        t.record(f"generate_total_time{sfx}", res["total_time"])
        t.record(f"generate{sfx}", res["total_time"])
        t.record(f"tokens_generated{sfx}", res["tokens_generated"])
        t.record(f"tokens_per_second{sfx}", res["tokens_per_second"])
        for k in ("itl_mean", "itl_p50", "itl_p95", "itl_max"):
            t.record(f"{k}{sfx}", res[k])

        state["raw"] = res["response"]
        state["answers"] = [extract_answer(res["response"])]
        print("Answer:", state["answers"][0])

    def _run_batch(self, state, ctx, g, contexts):
        t, sfx, dev = ctx.timing, ctx.sfx, ctx.devices["gen"]
        qs = state["questions"]
        ctxs = contexts if contexts is not None else [None] * len(qs)
        prompts = [g.build_prompt(q, c) for q, c in zip(qs, ctxs)]

        t.start(f"tokenize{sfx}")
        inputs = g.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if dev is not None:
            inputs = inputs.to(dev)
        t.end(f"tokenize{sfx}")

        if dev is not None:
            t.reset_gpu_peak(dev)
            t.capture_gpu_memory(f"gen_gpu_before{sfx}", dev)
        t.capture_all_gpu_memory(f"gen_gpu_all_before{sfx}")
        t.record(f"input_tokens_total{sfx}", inputs["input_ids"].shape[0] * inputs["input_ids"].shape[1])

        recorder = TTFTRecorder(inputs["input_ids"].shape[-1])
        ctx.mark("generate_start")
        t.start(f"generate{sfx}")
        with torch.no_grad():
            outputs = g.model.generate(
                **inputs,
                max_new_tokens=32,
                pad_token_id=g.tokenizer.pad_token_id,
                eos_token_id=g.tokenizer.eos_token_id,
                logits_processor=[recorder],
            )
        t.end(f"generate{sfx}")
        ctx.mark("generate_end")
        # per-step latency of the whole batch (every sequence advances one token per step)
        for k, v in itl_stats(recorder.step_times).items():
            t.record(f"{k}{sfx}", v)

        if dev is not None:
            t.capture_gpu_memory(f"gen_gpu_after{sfx}", dev)
        t.capture_all_gpu_memory(f"gen_gpu_all_after{sfx}")

        prompt_len = inputs["input_ids"].shape[-1]
        n_gen = sum(o.shape[-1] - prompt_len for o in outputs)
        gen_s = t.data[f"generate{sfx}"]
        t.record(f"tokens_generated{sfx}", n_gen)
        t.record(f"tokens_per_second{sfx}", n_gen / gen_s if gen_s > 0 else 0)

        t.start(f"decode{sfx}")
        state["answers"] = [
            extract_answer(g.tokenizer.decode(o[prompt_len:], skip_special_tokens=True))
            for o in outputs
        ]
        t.end(f"decode{sfx}")


@register_condition("answer_unchanged")
def answer_unchanged(state, ctx):
    """Stop iterating once every answer equals the previous round's answer."""
    prev = state.get("_prev_answers")
    state["_prev_answers"] = list(state["answers"])
    return prev is not None and prev == state["answers"]
