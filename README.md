# PII Redaction (MLX fork)

A Python package for redacting Personally Identifiable Information (PII) from text using Large Language Models.

> **This is a fork of [OpenPipe/pii-redaction](https://github.com/OpenPipe/pii-redaction)** that adds an
> **MLX backend** so the same models run natively on Apple Silicon via
> [oMLX](https://github.com/lmstudio-ai), with **no torch, CUDA, or GPU required**.
> The original `transformers` path is preserved and selected automatically when no
> oMLX server is reachable.
>
> See **[CHANGES-vs-upstream.md](CHANGES-vs-upstream.md)** for a full comparison
> against `OpenPipe/pii-redaction`.

## How the two backends work

| Backend | What it does | Needs |
| --- | --- | --- |
| `omlx` | Sends each document to an oMLX OpenAI-compatible server (`/v1/chat/completions`). oMLX runs the `PII-Redact-*-oQ4` MLX quants through **MLX** on Apple Silicon. | a running oMLX server, `requests` |
| `transformers` | The original OpenPipe path: loads the checkpoints locally with `transformers` + `torch` and generates with `model.generate`. | `transformers`, `torch` |

Backend selection is `auto` by default: if an oMLX server answers on the configured
base URL it is used, otherwise the code falls back to `transformers`.

## Installation

Core install (MLX / oMLX backend — no torch):

```bash
pip install pii-redact-mlx
```

Original local `transformers` (torch) backend as well:

```bash
pip install "pii-redact-mlx[transformers]"
```

Or install from source:

```bash
git clone https://github.com/<you>/pii-redact-mlx.git
cd pii-redact-mlx
pip install -e .            # core (MLX backend)
pip install -e ".[all]"     # both backends
```

## Running with MLX (oMLX)

Download the two models and serve them with oMLX, e.g.:

```bash
# models are addressed by their served name; the defaults match these ids
#   PII-Redact-Name, PII-Redact-General
omlx-cli serve --model-dir ~/.omlx/models --host 0.0.0.0 --port 8000
```

Point the tool at the server. Either set the environment variable:

```bash
export OMLX_BASE_URL=http://localhost:8000/v1
```

…or pass `--omlx-url` on the command line. With the server running, the default
`--backend auto` picks MLX automatically.

## Concurrency

Inference requests run in parallel. Every `(document, model)` pair is dispatched
to a thread pool, so both many documents and the two models per document are
processed at once. Results are reassembled in document order, so **output is
identical to the sequential path**.

- Default: **8** parallel requests for the `omlx` backend, **1** for
  `transformers` (a single torch model is not reentrant, so its `generate` calls
  are serialized behind a lock).
- Override with `--concurrency N` or the `PII_REDACT_CONCURRENCY` env var.
- `process-jsonl` reads lines in batches (`--batch-size`, default
  `max(16, concurrency * 4)`) and tags every message in a batch in one call,
  flushing after each batch.

The oMLX backend keeps one pooled `requests.Session` per thread, so connections
are reused across concurrent calls.

**Match the server:** oMLX caps parallelism with `scheduler.max_concurrent_requests`
(default 8; `--max-concurrent-requests` on the CLI, or the oMLX.app settings).
Client concurrency above that just queues server-side, so set `--concurrency` to
roughly the server's limit.

On the oMLX.app GUI the value is applied **at server start** — raise it, then
restart oMLX, and confirm it took (check `~/.omlx/settings.json`, or just watch
whether throughput keeps climbing with more client workers). Raising it from 8 to
32 on an M-series Mac roughly doubled throughput for short PII requests
(~4.7 → ~7.9 req/s) before plateauing — MLX generation is memory-bandwidth bound,
so past a certain point extra requests only add latency.

## Usage

### Command Line Interface

The package provides a command-line tool `pii-redact` with the following commands:

#### Process a JSONL dataset

For handling PII in JSONL files that contain messages (like conversation history):

```bash
pii-redact process-jsonl input.jsonl output.jsonl
```

Options:
- `--backend`: `auto` (default), `transformers`, or `omlx`
- `--omlx-url`: oMLX base URL, e.g. `http://localhost:8000/v1` (env: `OMLX_BASE_URL`)
- `--omlx-model-name`: oMLX model name for the person/organization model (default `PII-Redact-Name`)
- `--omlx-model-general`: oMLX model name for the general model (default `PII-Redact-General`)
- `--concurrency`: parallel inference requests (default 8 for omlx, 1 for transformers; env `PII_REDACT_CONCURRENCY`)
- `--batch-size`: JSONL lines per concurrent batch (default `max(16, concurrency * 4)`)
- `--max-tokens`: fixed generation budget per request (default 1024; env `PII_REDACT_MAX_TOKENS`)
- `--auto-max-tokens`: auto-size the generation budget from each input's length — **needed for documents longer than ~1024 tokens**, since the model echoes the input back with tags
- `--device`: Device for the `transformers` backend (e.g., cuda, cpu, mps)
- PII handling modes (mutually exclusive):
  - `--tag`: Keep PII content between XML tags (default) `<PII:type>content</PII:type>`
  - `--redact`: Replace PII with just an empty tag `<PII:type/>`
  - `--replace`: Replace PII values with fake data — **tags are dropped**, e.g. `Terri Romero`
- `--locale`: Locale for generating fake data (default: en_US, only used with --replace)

#### Process text files

For handling PII in plain text files (one document per line):

```bash
pii-redact process-text input.txt output.txt
```

Options: same as `process-jsonl` above.

#### Convert captured request traces to a redacted OpenAI JSONL

If you have a trace log (one captured chat request per line — e.g. a gateway/proxy
such as Plexus), this redacts every message and emits a standard OpenAI
chat-completions JSONL:

```bash
pii-redact convert-traces traces.jsonl redacted_openai.jsonl
```

The request payload is auto-detected in `transformed_request`, `raw_request`,
`request`, `body`, or `payload` (nested object or JSON-encoded string). Output is
one `{"messages": [...]}` object per line. Default mode is `--redact`.

Options:
- `--request-field`: field holding the chat request (default: auto-detect)
- `--keep-field`: copy a top-level record field into each output line (repeatable)
- `--auto-max-tokens` / `--max-tokens`: generation budget (see below)

Because a trace file repeats the same system prompt and conversation prefixes
across requests, identical message contents are **deduplicated and redacted once**,
then mapped back — often a >10x saving on real trace files.

#### List the models served by oMLX

```bash
pii-redact list-models --omlx-url http://localhost:8000/v1
```

#### Examples

Tag PII in text documents (default mode, auto backend):

```bash
pii-redact process-text emails.txt tagged_emails.txt
```

Force the MLX/oMLX backend:

```bash
pii-redact process-text emails.txt tagged_emails.txt \
  --backend omlx --omlx-url http://localhost:8000/v1
```

Redact PII completely:

```bash
pii-redact process-text emails.txt redacted_emails.txt --redact
```

Replace PII with fake data:

```bash
pii-redact process-text emails.txt anonymized_emails.txt --replace
```

Use a specific locale for fake data:

```bash
pii-redact process-text emails.txt anonymized_emails.txt --replace --locale=fr_FR
```

Process a JSONL dataset and redact PII:

```bash
pii-redact process-jsonl conversations.jsonl redacted_conversations.jsonl --redact
```

### Python API

```python
from pii_redaction import tag_pii_in_documents, clean_dataset, PIIHandlingMode

documents = [
    "My name is John Doe and my email is john.doe@example.com",
    "Call me at 555-123-4567 and ask for my SSN: 123-45-6789",
]

# Tag PII (default mode). Backend 'auto' uses oMLX/MLX when available.
tagged_documents = tag_pii_in_documents(documents, mode=PIIHandlingMode.TAG)

# Redact PII completely
redacted_documents = tag_pii_in_documents(documents, mode=PIIHandlingMode.REDACT)

# Replace PII with fake data
anonymized_documents = tag_pii_in_documents(
    documents,
    mode=PIIHandlingMode.REPLACE,
    locale="en_US",
)

# Force the MLX/oMLX backend explicitly, 16 requests in parallel
tagged = tag_pii_in_documents(
    documents,
    mode=PIIHandlingMode.TAG,
    backend="omlx",
    omlx_base_url="http://localhost:8000/v1",
    concurrency=16,
)

# Process a JSONL dataset
clean_dataset("input.jsonl", "output.jsonl", mode=PIIHandlingMode.TAG)
clean_dataset("input.jsonl", "redacted.jsonl", mode=PIIHandlingMode.REDACT)
clean_dataset(
    "input.jsonl",
    "anonymized.jsonl",
    mode=PIIHandlingMode.REPLACE,
    locale="en_US",
)
```

#### Environment variables

- `OMLX_BASE_URL` — default oMLX base URL (default `http://localhost:8000/v1`)
- `OMLX_API_KEY` — optional bearer token sent to the oMLX server
- `PII_REDACT_BACKEND` — default backend (`auto`, `transformers`, or `omlx`)
- `PII_REDACT_CONCURRENCY` — default number of parallel inference requests

#### Key Features

**Multiple PII handling options**:
   - **Tag PII**: Identify and keep PII with XML tags like `<PII:email_address>john.doe@example.com</PII:email_address>`
   - **Redact PII**: Replace PII with just an empty tag like `<PII:email_address/>`
   - **Replace PII**: Swap each PII value for realistic fake data, dropping the tags, e.g. `jane.smith@example.org`. (Note: unlike tag/redact, this mode emits no `<PII:...>` markup.)

**Pluggable backends**: run locally with `transformers`/`torch`, or on Apple
Silicon with MLX through an oMLX server — same API, same output.

**Customizable**: Choose from different locales for generating culturally appropriate fake data

**Consistent replacement**: When replacing PII with fake data, maintains consistency (same PII values are replaced with the same fake values)

## Supported PII Categories

The model can identify and tag the following PII categories:

- age: a person's age
- credit_card_info: a credit card number, expiration date, CCV, etc.
- nationality: a country when used to reference place of birth, residence, or citizenship
- date: a specific calendar date
- date_of_birth: a specific calendar date representing birth
- domain_name: a domain on the internet
- email_address: an email ID
- demographic_group: Anything that identifies race or ethnicity
- gender: a gender identifier
- personal_id: Any ID string like a national ID, subscriber number, etc.
- other_id: Any ID not associated with a person like an organization ID, database ID, etc.
- banking_number: a number associated with a bank account
- medical_condition: A diagnosis, treatment code or other information identifying a medical condition
- organization_name: name of an organization
- person_name: name of a person
- phone_number: a telephone number
- street_address: a physical address
- password: a secure string used for authentication
- secure_credential: any secure credential like an API key, private key, 2FA token
- religious_affiliation: anything that identifies religious affiliation

## Models & validation

The oMLX backend defaults to the **oQ4 MLX quants** — `PII-Redact-Name-oQ4` and
`PII-Redact-General-oQ4` (~726 MB each, vs ~2.5 GB full precision). Select others
with `--omlx-model-name` / `--omlx-model-general`.

`testdata/pii_test_set.jsonl` is a small labelled set (18 documents, synthetic
PII — `example.com` addresses and 555 numbers). Validate any configuration
against it:

```bash
python scripts/validate.py --omlx-api-key-file ~/.omlx/api_key
```

Measured on the oQ4 quants, with full precision as a reference on the same 18
documents (span precision/recall/F1 against the labels):

```
full precision   P=0.909  R=0.769  F1=0.833   (tp=20 fp=2 fn=6)
oQ4 quant        P=0.870  R=0.769  F1=0.816   (tp=20 fp=3 fn=6)
```

Recall is identical and both find the same 20 spans; the quant adds one spurious
span. Quant output varies slightly run-to-run even at temperature 0 (F1 observed
0.80–0.82), so a one-span gap is within noise. Both configurations miss the same
items — phone numbers (0/3) and a bank account number, and both label a
credit-card number `personal_id` — so those are the models' own tagging
behaviour, not quantisation damage.

## License

MIT (original work © OpenPipe; MLX fork modifications under the same license).
