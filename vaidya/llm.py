"""AyurParam language engine.

Three backends, chosen by the VAIDYA_BACKEND environment variable:
  llamacpp     - 4-bit GGUF build of AyurParam on CPU (hosted demo, college lab PCs)
  transformers - full bharatgenai/AyurParam on a GPU (Google Colab)
  none         - no model; the app still runs, with patient replies and
                 feedback taken straight from the faculty case file
  auto         - (default) GPU if present, else llamacpp, else none

AyurParam's own prompt format is used: '<context> ... <user> ... <assistant>'.
The model is loaded in a background thread so the interface opens at once.
"""
from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger("vaidya.llm")

HF_MODEL = os.getenv("VAIDYA_HF_MODEL", "bharatgenai/AyurParam")
GGUF_REPO = os.getenv("VAIDYA_GGUF_REPO", "A-Aryam/AyurParam-GGUF")
GGUF_FILE = os.getenv("VAIDYA_GGUF_FILE", "*Q4_K_M.gguf")
STOPS = ["<user>", "<context>", "<system_prompt>", "Doctor:", "\n\n\n"]


def build_prompt(user: str, context: str | None = None) -> str:
    if context:
        return f"<context> {context.strip()} <user> {user.strip()} <assistant>"
    return f"<user> {user.strip()} <assistant>"


def _clean(text: str) -> str:
    for s in STOPS:
        text = text.split(s)[0]
    return text.strip()


class Engine:
    """Thread-safe wrapper around whichever backend loads."""

    def __init__(self, backend: str | None = None):
        self.requested = (backend or os.getenv("VAIDYA_BACKEND", "auto")).lower()
        self.name = "none"
        self.status = "Not loaded"
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    # ---------- loading ----------
    def start(self) -> "Engine":
        if self.requested == "none":
            self.status = "Running without a language model (case-file mode)"
            self._ready.set()
            return self
        threading.Thread(target=self._load, daemon=True).start()
        return self

    def _load(self):
        order = [self.requested] if self.requested in ("llamacpp", "transformers") else self._auto_order()
        for name in order:
            try:
                self.status = f"Loading AyurParam ({name})... this takes a few minutes on first start"
                t0 = time.time()
                getattr(self, f"_load_{name}")()
                self.name = name
                self.status = f"AyurParam ready ({name}, loaded in {time.time() - t0:.0f} s)"
                log.info(self.status)
                break
            except Exception as e:  # noqa: BLE001 - try the next backend
                log.warning("Backend %s failed: %s", name, e)
                self.status = f"{name} unavailable: {e}"
        if self._model is None:
            self.status = "AyurParam could not be loaded here; running in case-file mode"
        self._ready.set()

    @staticmethod
    def _auto_order() -> list[str]:
        try:
            import torch
            if torch.cuda.is_available():
                return ["transformers", "llamacpp"]
        except Exception:  # noqa: BLE001
            pass
        return ["llamacpp"]

    def _load_llamacpp(self):
        from llama_cpp import Llama
        self._model = Llama.from_pretrained(repo_id=GGUF_REPO, filename=GGUF_FILE, n_ctx=2048,
                                            n_threads=max(1, os.cpu_count() or 2), verbose=False)

    def _load_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=False)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(HF_MODEL, trust_remote_code=True, torch_dtype=dtype,
                                                           device_map="auto", low_cpu_mem_usage=True)
        self._model.eval()

    # ---------- use ----------
    @property
    def ready(self) -> bool:
        return self._model is not None

    def wait(self, timeout: float | None = None) -> bool:
        self._ready.wait(timeout)
        return self.ready

    def generate(self, prompt: str, max_tokens: int = 120, temperature: float = 0.4) -> str | None:
        """Return the model's reply, or None if no model is available or it fails."""
        if not self.ready:
            return None
        try:
            with self._lock:
                if self.name == "llamacpp":
                    out = self._model(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9,
                                      top_k=50, stop=STOPS, echo=False)
                    return _clean(out["choices"][0]["text"])
                return _clean(self._generate_hf(prompt, max_tokens, temperature))
        except Exception as e:  # noqa: BLE001
            log.warning("Generation failed: %s", e)
            return None

    def _generate_hf(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import torch
        tok = self._tokenizer
        inputs = tok(prompt, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                       temperature=max(temperature, 0.01), top_p=0.9, top_k=50,
                                       eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class FakeEngine(Engine):
    """Stand-in used in tests: returns canned text so the pipeline can be checked without the model."""

    def __init__(self, reply: str = "Yes doctor, that is right."):
        super().__init__("none")
        self.reply = reply
        self._model = object()
        self.name = "fake"
        self.status = "Fake engine (tests)"
        self._ready.set()
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_tokens: int = 120, temperature: float = 0.4) -> str | None:
        self.prompts.append(prompt)
        return self.reply
