# generation.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, LogitsProcessor, BitsAndBytesConfig
import os
import time
import threading


class TTFTRecorder(LogitsProcessor):
    def __init__(self, input_length):
        self.input_length = input_length
        self.first_token_time = None

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] == self.input_length + 1 and self.first_token_time is None:
            torch.cuda.synchronize()
            self.first_token_time = time.time()
        return scores

class Generator:
    def __init__(self, model_path: str, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = torch.device(device)  # <-- normalize
        self.device = device
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            token=token,
        ).to(self.device)
        """

        if "70B" in model_path:
            device_map = "auto"
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            device_map = {"": device}
            quantization_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            token=token,
            quantization_config=quantization_config
        )

        devices = set(p.device for p in self.model.parameters())
        print("DEVICES",devices)
        #print(self.model.hf_device_map)

        #assert 1 == 2

        print(f"[Generator] model param device = {next(self.model.parameters()).device}")

    def build_prompt(self, question: str, context: str = None) -> str:
        # Optional: if you want the model to use RAG context but STILL answer-only,
        # inject context into the question block (keeps your formatting style).
        if context:
            question_text = (
                "Answer the question using the context.\n"
                f"Context:\n{context}\n\n"
                f"{question}"
            )
        else:
            question_text = question

        prompt_in_chat_format3 = [
            {
                "role": "system",
                "content": "Answer the question. Give a short and concise response.",
            },
            {
                "role": "user",
                "content": f"""Answer with only the direct answer. Do not provide explanations.
            Here are some example question-answer pairs to help guide your response format:
                
                Example 1:
                Question: when did the lion king start on broadway?
                Answer: October 15, 1997

                Example 2:
                Question: when was the movie the wizard of oz made?
                Answer: 1939

                Example 3:
                Question: when does the next apollo book come out?
                Answer: May 1, 2018

                Example 4:
                Question: what kind of dog is nana in snow dogs?
                Answer: Border Collie

                Example 5:
                Question: who has won the most college football national championships?
                Answer: Princeton

                ---
                Now here is the actual question you need to answer.

                Question: {question_text}
                -
                """,
                        },
                    ]

        # just using self.tokenizer now
        prompt = self.tokenizer.apply_chat_template(
            prompt_in_chat_format3,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt

    def build_iterative_reasoning_prompt(self, question: str, context: str = None) -> str:
        ctx_block = f"Context:\n{context}\n\n" if context else ""
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": (
                    "You are a reasoning assistant. Think through the question step by step using the context. "
                    "Identify key facts and entities. It's okay if your answer is partial or uncertain."
                ),
            },
            {
                "role": "user",
                "content": f"{ctx_block}Question: {question}\nReason through what you know:",
            },
        ]
        return self.tokenizer.apply_chat_template(
            prompt_in_chat_format,
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_iterative_final_prompt(self, question: str, context: str = None) -> str:
        ctx_block = f"Context:\n{context}\n\n" if context else ""
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": "Give only the final answer. One word, name, or short phrase. No explanation.",
            },
            {
                "role": "user",
                "content": f"{ctx_block}Question: {question}\nAnswer:",
            },
        ]
        return self.tokenizer.apply_chat_template(
            prompt_in_chat_format,
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def generate_with_ttft(self, question, context=None, max_new_tokens=32, prompt_type="standard"):
        if prompt_type == "iterative_reasoning":
            prompt = self.build_iterative_reasoning_prompt(question, context)
        elif prompt_type == "iterative_final":
            prompt = self.build_iterative_final_prompt(question, context)
        else:
            prompt = self.build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        input_length = inputs["input_ids"].shape[-1]
        recorder = TTFTRecorder(input_length)

        #print(f"\n{'='*50}")
        #print(f"[TTFT DEBUG] Prompt: {prompt[:100]}...")
        #print(f"[TTFT DEBUG] Input tokens: {input_length}")

        torch.cuda.synchronize()
        start_time = time.time()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=[recorder],
        )

        torch.cuda.synchronize()
        end_time = time.time()

        generated_ids = outputs[:, input_length:]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        tokens_generated = generated_ids.shape[-1]
        total_time = end_time - start_time
        ttft = (recorder.first_token_time - start_time) if recorder.first_token_time else 0
        decode_time = total_time - ttft

        # sanity checks
        ttft_fired = recorder.first_token_time is not None
        expected_steps = tokens_generated  # logits processor should fire once per token
        
        """
        print(f"[TTFT DEBUG] Recorder fired:       {ttft_fired}")
        print(f"[TTFT DEBUG] Output tokens:        {tokens_generated}")
        print(f"[TTFT DEBUG] Total output shape:   {outputs.shape}")
        print(f"[TTFT DEBUG] Response:             '{response}'")
        print(f"[TTFT DEBUG] ---")
        print(f"[TTFT DEBUG] start_time:           {start_time:.6f}")
        print(f"[TTFT DEBUG] first_token_time:     {recorder.first_token_time:.6f}" if ttft_fired else "[TTFT DEBUG] first_token_time:     NEVER SET")
        print(f"[TTFT DEBUG] end_time:             {end_time:.6f}")
        print(f"[TTFT DEBUG] ---")
        print(f"[TTFT DEBUG] TTFT:                 {ttft*1000:.2f}ms")
        print(f"[TTFT DEBUG] Decode time:          {decode_time*1000:.2f}ms")
        print(f"[TTFT DEBUG] Total time:           {total_time*1000:.2f}ms")
        print(f"[TTFT DEBUG] Tokens/sec:           {tokens_generated / decode_time if decode_time > 0 else 0:.2f}")
        print(f"[TTFT DEBUG] TTFT % of total:      {(ttft/total_time)*100:.1f}%" if total_time > 0 else "")
        print(f"{'='*50}\n")

        # sanity warnings
        if not ttft_fired:
            print("[TTFT WARNING] Recorder never fired — input_length boundary may be wrong")
        if ttft > total_time * 0.9:
            print("[TTFT WARNING] TTFT is >90% of total time — decode time suspiciously low")
        if ttft < 0.001:
            print("[TTFT WARNING] TTFT <1ms — may still be a warmup/cache issue")
        if tokens_generated == 0:
            print("[TTFT WARNING] No tokens generated — check eos_token_id or max_new_tokens")

        """

        return {
            "response": response,
            "ttft": ttft,
            "decode_time": decode_time,
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / decode_time if decode_time > 0 else 0,
        }

    def generate(self, question: str, context: str = None, max_new_tokens: int = 32):
        prompt = self.build_prompt(question, context)
        inputs = self.tokenize_prompt(prompt)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
        )

        # Return ONLY the completion (not the prompt echoed back)
        prompt_len = inputs["input_ids"].shape[-1]
        completion_ids = output_ids[0, prompt_len:]
        response = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        return response
    
    
