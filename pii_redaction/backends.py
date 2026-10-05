"""
Inference backends for PII redaction.

Two backends ship with the package:

* ``transformers`` -- the original torch/transformers path (CUDA, CPU, MPS).
  Loads the ``OpenPipe/Pii-Redact-*`` checkpoints locally and generates with
  ``model.generate``.

* ``omlx`` -- talks to an `oMLX <https://github.com/jundot/omlx>`_ server over
  its OpenAI-compatible API (``/v1/chat/completions``). oMLX runs the same
  checkpoints through Apple's MLX framework, so PII redaction works on Apple
  Silicon without torch or a discrete GPU. Models are addressed by their served
  name (``PII-Redact-Name`` / ``PII-Redact-General``) instead of a local path,
  and the server applies the chat template and runs generation.

Both backends expose the same tiny interface (``tag`` / ``tag_batch``), so the
rest of the package is backend-agnostic. Both are safe to call from multiple
threads (the oMLX backend keeps one pooled ``requests.Session`` per thread; the
transformers backend serializes ``generate`` with a lock, since a single torch
model is not reentrant).
"""

import os
import threading
from typing import List, Optional

#: Default base URL for the oMLX OpenAI-compatible API.
DEFAULT_OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://localhost:8000/v1")

#: Generation budget, matching the transformers backend's ``max_new_tokens``.
DEFAULT_MAX_NEW_TOKENS = 1024

#: Upper bound used when the generation budget is auto-sized per request.
MAX_AUTO_NEW_TOKENS = 16384

#: Characters-per-token estimate for auto-sizing the generation budget. The
#: redaction models echo the input back with tags, so output tokens ~= input
#: tokens; this keeps the budget proportional to the prompt length.
CHARS_PER_TOKEN = 3.0


def _auto_max_tokens(text: str) -> int:
    """Estimate a generation budget that can echo ``text`` back with tags."""
    return min(max(512, int(len(text) / CHARS_PER_TOKEN) + 256), MAX_AUTO_NEW_TOKENS)

#: Default number of in-flight requests when the oMLX backend is used.
DEFAULT_OMLX_CONCURRENCY = 8

#: Client read timeout for one /chat/completions call, overridable by env.
#:
#: The endpoint is NOT streamed, so this covers the ENTIRE generation, not the
#: gap between bytes.  These models echo their input, so a chunk of N tokens
#: produces roughly 1.5N -- and with several slots sharing one GPU each request
#: is correspondingly slower.  A 6000-token chunk can therefore legitimately
#: take far longer than the old 300s default, which surfaced as
#:     Read timed out. (read timeout=300.0)
#: and killed a run 61 minutes in, after it had already checkpointed 138 rows.
DEFAULT_OMLX_TIMEOUT = 300.0


def resolve_omlx_timeout() -> float:
    """Read ``OMLX_TIMEOUT`` at CALL time rather than at import time.

    Reading the environment in a module-level assignment freezes the value on
    first import, which makes the override depend on import ORDER: set the
    variable after ``backends`` has been imported and the assignment is
    silently missed, the 300s default comes back, and a run dies 61 minutes in
    with the very timeout error the override exists to prevent. Resolving it
    here means the env var works whether it is exported before the process
    starts or assigned in-process before the backend is constructed.
    """
    raw = os.environ.get("OMLX_TIMEOUT")
    if raw is None or not str(raw).strip():
        return DEFAULT_OMLX_TIMEOUT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_OMLX_TIMEOUT
    return value if value > 0 else DEFAULT_OMLX_TIMEOUT


class BackendError(RuntimeError):
    """Raised when a backend cannot be initialized or a generation call fails."""


def resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve the oMLX API key without ever requiring it on the command line.

    Precedence: an explicit argument, then ``OMLX_API_KEY``, then the stripped
    contents of the file named by ``OMLX_API_KEY_FILE``. Pointing at a file keeps
    the secret out of shell history, process listings, and chat.
    """
    if api_key:
        return api_key
    env_key = os.environ.get("OMLX_API_KEY")
    if env_key:
        return env_key
    key_file = os.environ.get("OMLX_API_KEY_FILE")
    if key_file:
        try:
            with open(key_file) as fh:
                return fh.read().strip() or None
        except OSError:
            return None
    return None


def auth_headers(api_key: Optional[str] = None) -> dict:
    """Return JSON + bearer auth headers for the oMLX API."""
    headers = {"Content-Type": "application/json"}
    key = resolve_api_key(api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


class InferenceBackend:
    """Minimal interface every backend implements."""

    name = "base"

    def tag(self, text: str) -> str:
        """Return ``text`` annotated with ``<PII:type>...</PII:type>`` tags."""
        raise NotImplementedError

    def tag_batch(self, texts: List[str]) -> List[str]:
        return [self.tag(t) for t in texts]


class TransformersBackend(InferenceBackend):
    """Local torch/transformers inference (the original OpenPipe path).

    A single torch model is not reentrant, so ``generate`` is guarded by a lock;
    concurrent callers are serialized rather than corrupting the model state.
    """

    name = "transformers"

    def __init__(self, model_path: str, device: Optional[str] = None, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise BackendError(
                "The 'transformers' backend requires torch and transformers. "
                "Install them with `pip install 'pii-redact-mlx[transformers]'`, "
                "or use the oMLX backend instead (`--backend omlx`)."
            ) from exc

        self._torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.padding_side = "left"
        self._lock = threading.Lock()
        self.max_new_tokens = max_new_tokens

        if device:
            self.model = self.model.to(device)
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def tag(self, text: str) -> str:
        torch = self._torch
        tokenizer = self.tokenizer

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        messages = [{"role": "user", "content": text}]

        encoded_input = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        input_ids = encoded_input["input_ids"].to(self.model.device)
        attention_mask = encoded_input["attention_mask"].to(self.model.device)

        # A single torch model is not safe to generate from concurrently.
        with self._lock, torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = encoded_input["input_ids"].size(1)
        generated_ids = outputs[0][input_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


class OMLXBackend(InferenceBackend):
    """MLX inference served by an oMLX OpenAI-compatible endpoint.

    The oMLX HTTP API is stateless; the chat template and generation both happen
    server-side, so this backend only has to forward the prompt and read the
    assistant message back. Each thread gets its own pooled ``requests.Session``
    so concurrent requests reuse connections without sharing session state.
    """

    name = "omlx"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS,
        concurrency: int = DEFAULT_OMLX_CONCURRENCY,
    ):
        try:
            import requests
            from requests.adapters import HTTPAdapter
        except ImportError as exc:  # pragma: no cover - depends on env
            raise BackendError(
                "The oMLX backend requires the 'requests' package. "
                "Install it with `pip install requests` (it is a core dependency "
                "of pii-redact-mlx)."
            ) from exc

        self._requests = requests
        self._HTTPAdapter = HTTPAdapter
        self.model = model
        self.base_url = (base_url or DEFAULT_OMLX_BASE_URL).rstrip("/")
        self.api_key = resolve_api_key(api_key)
        # Resolve here, not at import: an explicit argument wins, otherwise the
        # env var is read now (see resolve_omlx_timeout for why not at import).
        self.timeout = resolve_omlx_timeout() if timeout is None else float(timeout)
        self.max_tokens = max_tokens
        self.concurrency = max(1, int(concurrency))
        self._local = threading.local()

    @property
    def _headers(self):
        return auth_headers(self.api_key)

    def _session(self):
        """Return this thread's ``requests.Session``, creating it on first use."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._requests.Session()
            adapter = self._HTTPAdapter(
                pool_connections=self.concurrency,
                pool_maxsize=self.concurrency,
                max_retries=0,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def tag(self, text: str) -> str:
        url = f"{self.base_url}/chat/completions"
        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = _auto_max_tokens(text)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }

        try:
            response = self._session().post(
                url, json=payload, headers=self._headers, timeout=self.timeout
            )
        except Exception as exc:  # requests.exceptions.RequestException
            raise BackendError(
                f"Could not reach the oMLX server at {self.base_url}: {exc}. "
                "Is oMLX running (e.g. `omlx-cli serve ...`)?"
            ) from exc

        if response.status_code != 200:
            raise BackendError(
                f"oMLX server error {response.status_code} for model "
                f"{self.model!r}: {response.text[:500]}"
            )

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise BackendError(
                f"Unexpected response from oMLX server: {str(data)[:500]}"
            ) from exc


def omlx_server_available(
    base_url: Optional[str] = None, timeout: float = 2.0
) -> bool:
    """Return True if an oMLX server answers on ``base_url`` (GET /models)."""
    try:
        import requests
    except ImportError:
        return False

    url = (base_url or DEFAULT_OMLX_BASE_URL).rstrip("/") + "/models"
    try:
        response = requests.get(url, headers=auth_headers(), timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def list_omlx_models(base_url: Optional[str] = None, timeout: float = 5.0):
    """Return the list of model ids served by an oMLX server."""
    try:
        import requests
    except ImportError as exc:
        raise BackendError(
            "Listing oMLX models requires the 'requests' package."
        ) from exc

    url = (base_url or DEFAULT_OMLX_BASE_URL).rstrip("/") + "/models"
    try:
        response = requests.get(url, headers=auth_headers(), timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        raise BackendError(
            f"Could not list models from {url}: {exc}. If the server requires "
            "auth, pass --omlx-api-key-file <path> (or set OMLX_API_KEY_FILE)."
        ) from exc
    return [m["id"] for m in response.json().get("data", [])]
