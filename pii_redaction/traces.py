"""
Convert captured LLM API traces into a redacted OpenAI chat-completions JSONL.

Input: a JSONL file where each line is a captured request record. The chat
request may live in a top-level field (``transformed_request``, ``raw_request``,
...), either as a nested object or as a JSON-encoded string -- both are handled.
This is the shape produced by gateway/proxy trace logs (e.g. Plexus).

Output: one JSON object per line in the standard OpenAI messages format::

    {"messages": [{"role": "user", "content": "..."}, ...]}

Every message ``content`` string is passed through the PII redaction model.
Because a trace file repeats the same system prompt and conversation prefixes
across requests, identical contents are deduplicated and redacted **once**, then
mapped back -- the dominant cost saver on real trace files.
"""

import json
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .redactor import PIIHandlingMode, PIIRedactor

#: Top-level record fields checked (in order) for a chat request payload.
DEFAULT_REQUEST_FIELDS: Tuple[str, ...] = (
    "transformed_request",
    "raw_request",
    "request",
    "body",
    "payload",
)


def _as_request(value):
    """Return ``value`` as a dict if it is one or a JSON-encoded object."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    return value if isinstance(value, dict) else None


def extract_messages(record: dict, request_field: Optional[str] = None) -> Optional[list]:
    """Return the ``messages`` list from a trace ``record``, or None.

    ``request_field`` names the top-level field holding the request; when None,
    ``DEFAULT_REQUEST_FIELDS`` is tried, then ``record['messages']`` itself.
    """
    if request_field:
        candidates = [request_field]
    else:
        candidates = list(DEFAULT_REQUEST_FIELDS)

    for field in candidates:
        if field not in record:
            continue
        request = _as_request(record[field])
        if request is not None and isinstance(request.get("messages"), list):
            return request["messages"]

    if request_field is None and isinstance(record.get("messages"), list):
        return record["messages"]
    return None


def convert_traces(
    input_filename: str,
    output_filename: str,
    redactor: Optional[PIIRedactor] = None,
    request_field: Optional[str] = None,
    mode: PIIHandlingMode = PIIHandlingMode.REDACT,
    locale: str = "en_US",
    keep_fields: Sequence[str] = (),
    progress: bool = False,
    **redactor_kwargs,
) -> Dict[str, object]:
    """Convert a trace JSONL into a redacted OpenAI chat JSONL.

    Args:
        input_filename: Trace JSONL (one request record per line).
        output_filename: Destination OpenAI messages JSONL.
        redactor: Optional pre-built :class:`PIIRedactor`. If omitted, one is
            created from ``redactor_kwargs``.
        request_field: Field holding the chat request (auto-detected when None).
        mode: PII handling mode (TAG / REDACT / REPLACE).
        locale: Locale for fake data (REPLACE mode only).
        keep_fields: Extra top-level record fields to copy into each output line.
        progress: Show a progress bar over the inference calls.
        **redactor_kwargs: Passed to :class:`PIIRedactor` when ``redactor`` is None
            (e.g. ``backend=``, ``omlx_base_url=``, ``concurrency=``,
            ``auto_max_tokens=True``).

    Returns:
        A dict of statistics (record and message counts, unique documents, chars,
        elapsed seconds and throughput).
    """
    t_start = time.time()

    if redactor is None:
        redactor = PIIRedactor(**redactor_kwargs)

    # --- Pass 1: stream records, keep only the messages we need -----------------
    kept_records: List[dict] = []  # {"messages": [...], **kept_fields}
    unique_index: Dict[str, int] = {}
    unique_contents: List[str] = []

    records_in = 0
    skipped = 0
    non_str_content = 0

    with open(input_filename, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records_in += 1

            messages = extract_messages(record, request_field)
            if not messages:
                skipped += 1
                continue

            for message in messages:
                content = message.get("content")
                if isinstance(content, str) and content:
                    if content not in unique_index:
                        unique_index[content] = len(unique_contents)
                        unique_contents.append(content)
                elif content is not None:
                    non_str_content += 1  # e.g. multimodal lists: left untouched

            entry = {"messages": messages}
            for field in keep_fields:
                if field in record:
                    entry[field] = record[field]
            kept_records.append(entry)

    # --- Pass 2: redact each unique content exactly once ------------------------
    t_infer = time.time()
    if unique_contents:
        redacted = redactor.tag_pii_in_documents(
            unique_contents, mode=mode, locale=locale, progress=progress
        )
    else:
        redacted = []
    infer_seconds = time.time() - t_infer
    mapping = dict(zip(unique_contents, redacted))

    # --- Pass 3: reassemble and write ------------------------------------------
    tags = 0
    written = 0
    with open(output_filename, "w") as fout:
        for entry in kept_records:
            out_messages = []
            for message in entry["messages"]:
                content = message.get("content")
                if isinstance(content, str) and content in mapping:
                    new_content = mapping[content]
                    tags += new_content.count("<PII:")
                    if new_content != content:
                        message = dict(message)
                        message["content"] = new_content
                out_messages.append(message)

            out = {"messages": out_messages}
            for field in keep_fields:
                if field in entry:
                    out[field] = entry[field]
            fout.write(json.dumps(out) + "\n")
            written += 1

    unique_chars = sum(len(c) for c in unique_contents)
    total_seconds = time.time() - t_start
    return {
        "records_in": records_in,
        "records_out": written,
        "records_skipped": skipped,
        "messages_in": sum(len(e["messages"]) for e in kept_records),
        "unique_contents": len(unique_contents),
        "unique_chars": unique_chars,
        "non_str_content": non_str_content,
        "tags_inserted": tags,
        "infer_seconds": round(infer_seconds, 2),
        "total_seconds": round(total_seconds, 2),
        # Input tokens are roughly unique_chars/3.5; ~10x that for a generation
        # pair (tagged echo) across the two models, so throughput is a rough
        # proxy rather than a precise token count.
        "unique_chars_per_sec": round(unique_chars / infer_seconds) if infer_seconds else None,
    }
