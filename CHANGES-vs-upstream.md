# Differences from upstream (`OpenPipe/pii-redaction`)

This repository is a fork of [`OpenPipe/pii-redaction`](https://github.com/OpenPipe/pii-redaction).
It keeps the original redaction logic and prompt contract intact, and changes
**how inference is run** and **what the CLI can be pointed at**.

Upstream is a single-backend, single-machine tool: it imports `torch` and
`transformers` at module load, loads two hard-wired checkpoints, and calls
`model.generate` sequentially. The fork makes the backend pluggable so the same
redaction models can run through a local **MLX** server on Apple Silicon, adds
concurrent inference, and adds a converter that turns captured request traces
into a redacted OpenAI chat JSONL.

Nothing about the PII taxonomy, the `<PII:type>...</PII:type>` tag format, or the
three handling modes changed — output is byte-compatible with upstream for the
same inputs and mode.

---

## At a glance

| Area | Upstream | This fork |
|---|---|---|
| Inference | `transformers` + `torch` only | Pluggable: `transformers` **or** `omlx` (OpenAI-compatible HTTP) |
| GPU requirement | CUDA / CPU / MPS via torch | None for the `omlx` backend — server runs the MLX weights |
| Backend selection | implicit | `--backend {auto,transformers,omlx}`, env `PII_REDACT_BACKEND` |
| Concurrency | sequential | `ThreadPoolExecutor`, `--concurrency` (default 8 omlx / 1 transformers) |
| Generation budget | fixed `max_new_tokens=1024` | `--max-tokens` or `--auto-max-tokens` (scales with input) |
| Trace conversion | — | `convert-traces` command (trace JSONL → OpenAI messages JSONL) |
| Model addressing | hard-wired HF repo ids | `--omlx-model-name` / `--omlx-model-general` (served names) |
| Packaging | torch/transformers hard deps | `[transformers]` optional extra; core = `requests`, `tqdm`, `faker` |
| Version | 0.1.x | 0.2.0 |

---

## 1. Pluggable inference backends (`pii_redaction/backends.py`, new)

Upstream hard-codes the torch path in `redactor.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
...
self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
outputs = self.model.generate(...)
```

That means importing the package fails without torch, and the only way to run
inference is with the weights loaded in-process.

The fork introduces an `InferenceBackend` interface with two implementations:

- **`TransformersBackend`** — unchanged behaviour, but `torch`/`transformers` are
  imported **lazily inside the constructor**. Importing the package no longer
  requires torch, and a missing torch produces a clear error that names the
  install extra rather than an `ImportError` traceback.
- **`OMLXBackend`** — talks to an [oMLX](https://github.com/lmstudio-ai) server
  over its OpenAI-compatible API (`POST /v1/chat/completions`). The server loads
  the same `OpenPipe/PII-Redact-*` checkpoints through **MLX**, applies the chat
  template, and runs generation. The client is pure `requests`: no torch, no
  CUDA, no local weights. Models are addressed by their **served name**
  (`PII-Redact-Name`, `PII-Redact-General`) instead of a local path.

`PIIRedactor` gained a `backend=` argument accepting `"auto"`, `"transformers"`,
or `"omlx"`. `"auto"` prefers oMLX when `GET /models` answers, and falls back to
transformers otherwise — so the heavy path is never taken silently.

The redaction logic itself (span merging, the three modes, tag emission) is
untouched; it now consumes whatever the backend returns.

## 2. Concurrent inference

Upstream processes documents one at a time. The fork dispatches work across a
`ThreadPoolExecutor` (one task per document, or one per `(document, model)` pair)
and reassembles results **in document order**, so output is identical to the
sequential run — verified byte-for-byte on a 40-record JSONL.

Two correctness details matter for threads:

- The oMLX backend keeps one pooled `requests.Session` **per thread**
  (`threading.local`), with the connection pool sized to the concurrency. Sharing
  one `Session` across threads is not safe.
- A single torch model is not reentrant, so `TransformersBackend.generate` is
  guarded by a `threading.Lock`; concurrent callers serialize instead of
  corrupting model state. Concurrency therefore defaults to `1` for the
  transformers backend and `8` for oMLX.

`clean_dataset` reads its JSONL in batches (`--batch-size`, default
`max(16, concurrency * 4)`) and issues one concurrent call per batch, flushing
each batch — so memory stays bounded on large files and progress survives a
crash.

## 3. Auto-sized generation budget

Upstream caps generation at `max_new_tokens=1024`. Because the redaction model
**echoes the input back with tags**, a document longer than ~1024 tokens comes
back **truncated mid-sentence** — silently. On the trace corpus this fork was
built for (messages up to ~39 KB), that loses most of the document.

The fork adds:

- `--max-tokens N` — an explicit budget.
- `--auto-max-tokens` — estimate the budget from the input length
  (`len(text) / 3.0 + 256`, floored at 512, capped at 16384), so long documents
  are not cut off while short ones don't over-allocate.

Default remains 1024, matching upstream, when neither flag is given.

## 3a. File-based API key

oMLX can be configured to require a bearer token. To avoid putting the secret on
the command line (where it lands in shell history and `ps` output), the CLI reads
it from a **file**:

```
printf '%s' "$KEY" > ~/.omlx/api_key && chmod 600 ~/.omlx/api_key
pii-redact list-models --omlx-url http://localhost:27473/v1 --omlx-api-key-file ~/.omlx/api_key
```

Resolution order is `--omlx-api-key-file` (→ `OMLX_API_KEY_FILE`) → `OMLX_API_KEY`
→ none. The key file is a plain path; the key value is never an argv entry.

## 4. `convert-traces` — trace JSONL → redacted OpenAI JSONL

New command for the pipeline this fork was written for: a gateway/proxy trace log
(one captured chat request per line, e.g. from Plexus) becomes a redacted
**OpenAI chat-completions JSONL**.

```
pii-redact convert-traces traces.jsonl redacted.jsonl \
  --backend omlx --omlx-url http://localhost:27473/v1 \
  --auto-max-tokens --concurrency 8
```

It auto-detects the request payload in `transformed_request` / `raw_request` /
`request` / `body` / `payload` (nested object **or** JSON-encoded string), or
takes `--request-field`. Non-string content (e.g. multimodal part lists, and
`tool_calls` messages with `content: null`) is passed through untouched.

**Deduplication (inference-side only).** A trace file repeats the same system
prompt and conversation prefixes across requests — on the target corpus,
18,201 messages reduced to 1,666 unique contents (10.9× duplication). Identical
content strings are redacted **once** and mapped back. This is purely an
inference optimisation: **every record and every message is still written to the
output**, in order. No turns are dropped. The stats dict reports
`unique_contents` alongside `messages_in` / `records_out` so the ratio is visible.

## 5. CLI changes

Added flags: `--backend`, `--omlx-url`, `--omlx-model-name`,
`--omlx-model-general`, `--omlx-api-key`, `--concurrency`, `--batch-size`,
`--max-tokens`, `--auto-max-tokens`, `--request-field`, `--keep-field`.
New subcommand: `list-models` (query an oMLX server for served model ids).
New subcommand: `convert-traces`.

`--keep-field` copies extra top-level trace fields into each output line, so
useful metadata (model, timestamp, id) survives the conversion.

## 6. Packaging

Upstream requires torch and transformers unconditionally. The fork moves them
behind an optional extra:

```bash
pip install pii-redact-mlx                 # oMLX backend + all CLI commands
pip install 'pii-redact-mlx[transformers]' # adds torch + transformers
```

Core dependencies are now `requests`, `tqdm`, `faker`. `PIIRedactor`,
`omlx_server_available`, `list_omlx_models`, and `convert_traces` are exported
from the package root.

---

## 7. Default models: the oQ4 MLX quants

The oMLX backend defaults to the 4-bit MLX quants of the same checkpoints —
`PII-Redact-Name-oQ4` and `PII-Redact-General-oQ4` (~726 MB each, vs ~2.5 GB full
precision), from `gwyntel/PII-Redact-*-oQ4` on HuggingFace. Full precision stays
reachable via `--omlx-model-name` / `--omlx-model-general`.

oMLX ids are the **leaf directory name** under `model_dir`, so a model at
`~/.omlx/models/PII-Redact-Name-oQ4` is served as `PII-Redact-Name-oQ4`
(and one nested at `gwyntel/<name>` still serves as `<name>` — the `gwyntel/`
prefix is not part of the id).

## 8. Validation

`testdata/pii_test_set.jsonl` — 18 documents carrying ground-truth PII spans
(synthetic: `example.com`, 555 numbers). `scripts/validate.py` runs a configuration
in TAG mode and scores its spans against those labels (overall + per tag).

Measured on the oQ4 quants with full precision as a reference on the same set:

```
full precision   P=0.909  R=0.769  F1=0.833   (tp=20 fp=2 fn=6)
oQ4 quant        P=0.870  R=0.769  F1=0.816   (tp=20 fp=3 fn=6)
```

Identical recall; the quant adds one spurious span (and quant output varies
run-to-run at temperature 0, F1 ~0.80-0.82, so one span is within noise). Both
configs miss the same items — phone numbers, a bank account number — and both
label a credit-card number `personal_id`, so those are the models' tagging
behaviour rather than quantisation damage.

## Configuration

| Setting | Env var | CLI flag | Default |
|---|---|---|---|
| oMLX base URL | `OMLX_BASE_URL` | `--omlx-url` | `http://localhost:8000/v1` |
| oMLX API key | `OMLX_API_KEY` / `OMLX_API_KEY_FILE` | `--omlx-api-key-file` | none |
| Backend | `PII_REDACT_BACKEND` | `--backend` | `auto` |
| Concurrency | `PII_REDACT_CONCURRENCY` | `--concurrency` | 8 (omlx) / 1 (transformers) |

**Note on ports:** the package default is `http://localhost:8000/v1` (the oMLX
default). A server on a non-default port — e.g. oMLX.app's `27473` — must be
passed explicitly via `--omlx-url` or `OMLX_BASE_URL`.

## Measured behaviour

- Client concurrency above the server's `scheduler.max_concurrent_requests` only
  queues server-side. Raising the server cap from 8 → 32 roughly doubled
  short-request throughput (≈4.7 → ≈7.9 req/s) on the test machine.
- Throughput is **memory-bound, not request-bound**, for long documents: memory
  scales with the *total in-flight characters*, so a high request count over 6 KB
  messages can exhaust RAM well before the server's request cap is reached. Cap
  long-document workloads by in-flight characters and prefer a moderate
  `--concurrency`.

## Mode semantics (verified against the code)

The three modes operate on the model's tagged output as follows (checked by
calling `apply_tags` directly, no inference required):

| Mode | Result for a name + email | Keeps tags? |
|---|---|---|
| `--tag` | `<PII:person_name>John Smith</PII:person_name> … jane.doe@example.com` inside tags | yes |
| `--redact` | `<PII:person_name/>` … `<PII:email_address/>` | yes |
| `--replace` | `Terri Romero` … `anthony03@example.org` | **no** |

`--replace` substitutes fake values and **drops the markup** — it does not emit
`<PII:type>fake</PII:type>`. Fake values are stable per original value within a
run (`FakePIIGenerator` remembers the mapping), so the same person keeps the same
fake name throughout.

## Known limitations

- Concurrency is bounded by the oMLX server's request scheduler, not the client.
- Deduplication is exact-string only (no near-duplicate detection).
- The `omlx` backend does not stream; each request is buffered.
- `--replace` cannot be combined with tag retention: there is no mode that emits
  fake values *inside* `<PII:...>` markup. Use `--tag` if you need labelled spans.
- Span merging and the PII taxonomy are inherited from upstream and unchanged.
