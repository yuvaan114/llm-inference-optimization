import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmoptimization.config import settings

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"          # NVIDIA GPU (the H200 later)
    if torch.backends.mps.is_available():
        return "mps"           # Apple Silicon GPU (your MacBook)
    return "cpu"


class InferenceEngine:
    """Owns the loaded model + tokenizer. Loaded once, reused for every request."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.ready = False

    def load(self):
        self.device = settings.device or _pick_device()
        logger.info("Loading %s on %s ...", settings.model_name, self.device)
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        self.model.eval()  # inference mode: disables training-only behavior

        self.ready = True
        logger.info("Model loaded in %.1fs", time.time() - t0)

    @torch.inference_mode()  # no gradient tracking -> less memory, faster
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
        # 1. Wrap the prompt in the model's chat format
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 2. Text -> token IDs (tensors), moved onto the device
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        prompt_tokens = inputs.input_ids.shape[1]

        # 3. Run prefill + decode (blocking, all tokens at once)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)  # greedy

        t0 = time.time()
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        # 4. Keep only the newly generated tokens, decode back to text
        gen_ids = output_ids[0][prompt_tokens:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        completion_tokens = int(gen_ids.shape[0])

        return {
            "text": completion,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_s": round(elapsed, 3),
            "tokens_per_s": round(completion_tokens / elapsed, 2) if elapsed > 0 else None,
        }


engine = InferenceEngine()  # single shared instance