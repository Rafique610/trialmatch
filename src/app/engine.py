"""Model engine for the app: loads the fine-tuned target once, the draft on first use,
and runs one prompt through the chosen decoder (our own TRIA-9 / TRIA-11 code, pure greedy).
"""

import hashlib
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings
from src.data.prompt import SYSTEM, user_message
from src.kvcache.decode import greedy_cached, greedy_no_cache
from src.kvcache.model import Runner
from src.speculative.decode import speculative_greedy

MAX_NEW_TOKENS = 64


class Engine:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(settings.target_model_id)
        base = AutoModelForCausalLM.from_pretrained(settings.target_model_id, dtype=torch.float16, device_map="cuda")
        from peft import PeftModel

        local = settings.checkpoints_dir / "final"
        self.model = PeftModel.from_pretrained(base, local if local.exists() else settings.adapter_repo_id).merge_and_unload().eval()
        self.target = Runner(self.model)
        self._draft: Runner | None = None
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.lock = threading.Lock()  # one GPU: one request decodes at a time
        weights = local / "adapter_model.safetensors"
        self.provenance = {
            "base_model": settings.target_model_id, "adapter": settings.adapter_repo_id,
            "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()[:16] if weights.exists() else None,
            "draft_model": settings.draft_model_id, "precision": "fp16", "decoding": "greedy (argmax, no repetition penalty)",
            "gpu": torch.cuda.get_device_name(0),
        }

    @property
    def draft(self) -> Runner:
        if self._draft is None:
            m = AutoModelForCausalLM.from_pretrained(settings.model_source(settings.draft_model_id), dtype=torch.float16, device_map="cuda")
            self._draft = Runner(m.eval())
        return self._draft

    @property
    def draft_loaded(self) -> bool:
        return self._draft is not None

    def prompt_ids(self, sentences: list[str], criterion_type: str, criterion_text: str) -> torch.Tensor:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(sentences, criterion_type, criterion_text)}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return torch.tensor(self.tok(text, add_special_tokens=False)["input_ids"], device="cuda")

    def decode(self, ids: torch.Tensor, mode: str, gamma: int = 2) -> dict:
        """Returns the answer text plus measured timing for this one prompt."""
        with self.lock:
            torch.cuda.reset_peak_memory_stats()
            if mode == "speculative":
                r = speculative_greedy(self.target, self.draft, ids, MAX_NEW_TOKENS, gamma, eos=self.eos)
                decode_s = r["total_s"] - r["prefill_s"]
            elif mode == "no_cache":
                r = greedy_no_cache(self.target, ids, MAX_NEW_TOKENS, eos=self.eos)
                decode_s = r["decode_s"]
            else:
                r = greedy_cached(self.target, ids, MAX_NEW_TOKENS, eos=self.eos)
                decode_s = r["decode_s"]
            peak = torch.cuda.max_memory_allocated() / 1024**3
        tokens = [t for t in r["tokens"] if t != self.eos]
        n = len(r["tokens"])
        out = {"text": self.tok.decode(tokens), "tokens": r["tokens"], "prompt_tokens": int(ids.shape[0]), "new_tokens": n,
               "prefill_ms": round(r["prefill_s"] * 1000, 1), "decode_ms": round(decode_s * 1000, 1),
               "tok_s": round(n / (r["prefill_s"] + decode_s), 2), "peak_gpu_gb": round(peak, 3)}
        if mode == "speculative":
            proposed = sum(x["proposed"] for x in r["rounds"])
            accepted = sum(x["accepted"] for x in r["rounds"])
            out |= {"gamma": gamma, "rounds": len(r["rounds"]), "acceptance_rate": round(accepted / proposed, 3) if proposed else None}
        return out
