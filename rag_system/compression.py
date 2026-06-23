# compression.py
from llmlingua import PromptCompressor
from selective_context import SelectiveContext
from transformers import AutoModel
import torch


FORCE_TOKENS = ["\n", ".", "!", "?", ","]
CHUNK_END_TOKENS = [".", "\n"]

class Compressor:
    def __init__(self, method, rate, torch_device="cuda:0"):
        self.method = method
        self.rate = rate

        # allow str or torch.device
        self.torch_device = torch_device
        if isinstance(self.torch_device, str):
            self.torch_device = torch.device(self.torch_device)

        if method == "llmlingua2":
            self.engine = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                llmlingua2_config={"max_force_token": 1024},
            )
            if not hasattr(self.engine, "max_force_token"):
                self.engine.max_force_token = 1024

            # best-effort: if underlying model exists, move it
            if hasattr(self.engine, "model"):
                try:
                    self.engine.model = self.engine.model.to(self.torch_device)
                except Exception:
                    pass

        elif method == "longllmlingua":
            # was: device = "cuda:2"
            device = str(self.torch_device)
            self.engine = PromptCompressor(
                model_name="meta-llama/Llama-3.2-1B-Instruct",
                device_map=device,
            )
            print(next(self.engine.model.parameters()).device)

        elif method == "sc":
            self.engine = SelectiveContext(model_type="gpt2", lang="en")
            # (SelectiveContext often handles device internally; leaving unchanged)

        elif method == "provence":
            # Provence exposes a custom .process(question, context, ...) API via trust_remote_code
            self.engine = AutoModel.from_pretrained(
                "naver/provence-reranker-debertav3-v1",
                trust_remote_code=True,
            )
            # was: device = "cuda:2"
            self.engine = self.engine.to(self.torch_device)
            self.engine.eval()

        else:
            self.engine = None

        # ---- PRINT DEVICE INFO ----
        self._print_engine_device()

    def _print_engine_device(self):
        if self.engine is None:
            print("[Compressor] No compression engine.")
            return

        # LLMLingua PromptCompressor case (wrapper that has .model)
        if hasattr(self.engine, "model"):
            if hasattr(self.engine, "device"):
                print(f"[Compressor] engine.device = {self.engine.device}")

            try:
                param = next(self.engine.model.parameters(), None)
                if param is not None:
                    print(f"[Compressor] model param device = {param.device}")
            except Exception as e:
                print(f"[Compressor] Could not read model parameters device: {e}")

            if hasattr(self.engine.model, "hf_device_map"):
                print(f"[Compressor] hf_device_map = {self.engine.model.hf_device_map}")

            return

        # Plain torch module case (Provence AutoModel, etc.)
        try:
            param = next(self.engine.parameters(), None)
            if param is not None:
                print(f"[Compressor] model param device = {param.device}")
            else:
                print(f"[Compressor] engine type = {type(self.engine)} (no params?)")
        except Exception as e:
            print(f"[Compressor] Could not read engine parameters device: {e}")

    def compress(self, context, *, question=None, instruction=None):
        if self.engine is None:
            return context

        # SelectiveContext
        if self.method == "sc":
            # GPT-2 max is 1024 tokens; truncate before passing in
            tok = self.engine.tokenizer
            ids = tok.encode(context)
            if len(ids) > 1024:
                context = tok.decode(ids[:1024])
            context, _ = self.engine(context, reduce_ratio=self.rate)
            return context

        # LongLLMLingua (v1-style API)
        if self.method == "longllmlingua":
            instruction = "Answer the question based on the provided context."
            question = "" if question is None else str(question)

            out = self.engine.compress_prompt(
                context,
                instruction=instruction,
                question=question,
                rate=self.rate,
                condition_compare=False,
            )

            compressed = out["compressed_prompt"] if isinstance(out, dict) else out
            if isinstance(compressed, list):
                compressed = "\n".join(compressed)
            return compressed

        # LLMLingua2 API
        if self.method == "llmlingua2":
            out = self.engine.compress_prompt_llmlingua2(
                context,
                rate=self.rate,
                force_tokens=FORCE_TOKENS,
                chunk_end_tokens=CHUNK_END_TOKENS,
            )
            return out["compressed_prompt"]

        # Provence (context pruning; requires question)
        if self.method == "provence":
            if question is None:
                return context

            threshold = float(self.rate)
            threshold = max(0.0, min(1.0, threshold))

            out = self.engine.process(
                str(question),
                str(context),
                threshold=threshold,
                always_select_title=False,
            )

            pruned = out.get("pruned_context", context) if isinstance(out, dict) else context

            if isinstance(pruned, list):
                pruned = "\n".join(map(str, pruned))

            return pruned

        return context
