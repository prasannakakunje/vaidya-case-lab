"""AyurParam language engine.

Backends, chosen by the VAIDYA_BACKEND environment variable:
  zerogpu      - full bharatgenai/AyurParam on a Hugging Face ZeroGPU Space
                 (GPU attached only while a reply is generated)
  llamacpp     - 4-bit GGUF build of AyurParam on CPU (CPU Space, college lab PCs)
  transformers - full bharatgenai/AyurParam on a GPU (Google Colab)
  none         - no model; the app still runs, with patient replies and
                 feedback taken straight from the faculty case file
  auto         - (default) ZeroGPU on a ZeroGPU Space, else a local GPU, else llamacpp, else none

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
ON_ZEROGPU = os.getenv("SPACES_ZERO_GPU", "").lower() in ("true", "1")
REPETITION_PENALTY = float(os.getenv("VAIDYA_REPETITION_PENALTY", "1.2"))


def is_repetitive(text: str, n: int = 3, limit: int = 3) -> bool:
    """True if any run of `n` words occurs more than `limit` times: the model is looping."""
    import re as _re
    w = _re.findall(r"\w+", text.lower())
    counts: dict = {}
    for i in range(len(w) - n + 1):
        key = tuple(w[i:i + n])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > limit:
            return True
    return False


def cut_repetition(text: str, n: int = 3) -> str:
    """Keep the text up to the point where it starts repeating itself."""
    import re as _re
    tokens = list(_re.finditer(r"\w+", text))
    seen: dict = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(t.group().lower() for t in tokens[i:i + n])
        if key in seen and seen[key] < i:
            return text[:tokens[i].start()].rstrip(" ,;") + " …"
        seen.setdefault(key, i)
    return text

# ---------- ZeroGPU: the decorated function must live at module level ----------
try:  # the 'spaces' package exists only on Hugging Face Spaces
    import spaces  # type: ignore
    _gpu = spaces.GPU(duration=60)
except Exception:  # noqa: BLE001
    def _gpu(fn):
        return fn

_HF: dict = {"model": None, "tok": None}


@_gpu
def _zerogpu_generate(prompt: str, max_tokens: int, temperature: float) -> str:
    import torch
    tok, model = _HF["tok"], _HF["model"]
    inputs = tok(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    return _hf_generate(tok, model, inputs, max_tokens, temperature)


def _hf_generate(tok, model, inputs, max_tokens: int, temperature: float) -> str:
    """Generate with transformers. AyurParam's own model code can fail with the KV cache on some
    transformers versions, so retry once without the cache before giving up."""
    import torch
    settings = dict(max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 0.01),
                    top_p=0.9, top_k=50, repetition_penalty=REPETITION_PENALTY, no_repeat_ngram_size=4,
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    with torch.no_grad():
        try:
            out = model.generate(**inputs, **settings)
        except Exception as e:  # noqa: BLE001
            log.warning("generate() failed (%s); retrying without KV cache", e)
            out = model.generate(**inputs, use_cache=False, **settings)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log.info("AyurParam raw output: %r", text[:300])
    return text


def build_prompt(user: str, context: str | None = None) -> str:
    if context:
        return f"<context> {context.strip()} <user> {user.strip()} <assistant>"
    return f"<user> {user.strip()} <assistant>"


def _clean(text: str) -> str:
    text = text.strip()
    for marker in ("<assistant>", "Patient:", "Answer:"):
        if text.startswith(marker):
            text = text[len(marker):].strip()
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
        self.last_error = ""

    # ---------- loading ----------
    def start(self) -> "Engine":
        if self.requested == "none":
            self.status = "Running without a language model (case-file mode)"
            self._ready.set()
            return self
        if self.requested == "zerogpu" or (self.requested == "auto" and ON_ZEROGPU):
            self._load()  # ZeroGPU needs the model placed on 'cuda' while the Space starts
        else:
            threading.Thread(target=self._load, daemon=True).start()
        return self

    def _load(self):
        order = [self.requested] if self.requested in ("zerogpu", "llamacpp", "transformers") else self._auto_order()
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
        if ON_ZEROGPU:
            return ["zerogpu", "llamacpp"]
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

    def _load_zerogpu(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.to("cuda").eval()
        _HF["tok"], _HF["model"] = tok, model
        self._tokenizer, self._model = tok, model

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
                if self.name == "zerogpu":
                    return _clean(_zerogpu_generate(prompt, max_tokens, temperature)) or None
                if self.name == "llamacpp":
                    out = self._model(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9,
                                      top_k=50, repeat_penalty=REPETITION_PENALTY, stop=STOPS, echo=False)
                    return _clean(out["choices"][0]["text"])
                return _clean(self._generate_hf(prompt, max_tokens, temperature)) or None
        except Exception as e:  # noqa: BLE001
            log.warning("Generation failed: %s: %s", type(e).__name__, e)
            self.last_error = f"{type(e).__name__}: {e}"
            return None

    def _generate_hf(self, prompt: str, max_tokens: int, temperature: float) -> str:
        tok = self._tokenizer
        inputs = tok(prompt, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        return _hf_generate(tok, self._model, inputs, max_tokens, temperature)


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
