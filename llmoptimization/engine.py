import logging
import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from llmoptimization.config import settings

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
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
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        self.model.eval()

        self.ready = True
        logger.info("Model loaded in %.1fs", time.time() - t0)

    def _prepare(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        return inputs

    def _sampling_kwargs(self, max_new_tokens: int, temperature: float) -> dict:
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        return kw

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
        inputs = self._prepare(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        t0 = time.time()
        output_ids = self.model.generate(**inputs, **self._sampling_kwargs(max_new_tokens, temperature))
        elapsed = time.time() - t0

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

    @torch.inference_mode()
    def generate_stream(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
        """Yields token chunks as they're generated, then a final stats event."""
        inputs = self._prepare(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(**inputs, streamer=streamer,
                          **self._sampling_kwargs(max_new_tokens, temperature))

        # model.generate runs on a background thread and pushes tokens into `streamer`
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        t_request = time.time()
        thread.start()

        ttft = None
        last_time = None
        tpot_sum = 0.0
        n = 0

        for chunk in streamer:
            now = time.time()
            if ttft is None:
                ttft = now - t_request          # first token -> TTFT
            else:
                tpot_sum += now - last_time      # gap between tokens -> TPOT
            last_time = now
            n += 1
            yield {"type": "token", "text": chunk}

        thread.join()
        total = time.time() - t_request
        tpot = (tpot_sum / (n - 1)) if n > 1 else None

        yield {
            "type": "done",
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "tpot_s": round(tpot, 4) if tpot is not None else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n,
            "tokens_per_s": round(n / total, 2) if total > 0 else None,
        }


engine = InferenceEngine()