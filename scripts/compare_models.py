#!/usr/bin/env python
"""
A/B compare two PII-redaction model configurations on the same documents.

Typical use: full-precision vs quantized (e.g. oQ4) weights.

    python scripts/compare_models.py --input sample.jsonl --limit 20 \
        --baseline-name PII-Redact-Name --baseline-general PII-Redact-General \
        --candidate-name PII-Redact-Name-oQ4 --candidate-general PII-Redact-General-oQ4 \
        --omlx-url http://localhost:27473/v1 --out reports/quant-ab.md

Both configurations run in TAG mode on identical inputs; results are compared
structurally (extracted ``(tag, text)`` spans) and textually.

Reported metrics (candidate vs baseline):
  * exact-match rate            -- fraction of documents whose output is byte-identical
  * span precision / recall / F1 -- candidate spans vs baseline spans (baseline = truth)
  * span Jaccard                -- intersection / union of extracted spans
  * tag-type agreement          -- per-document multiset of PII types
  * char delta                  -- mean absolute output length difference
"""

import argparse
import json
import sys
from collections import Counter
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, ".")

from pii_redaction.redactor import PIIRedactor, PIIHandlingMode, parse_tagged_string


def load_documents(path: str, limit: int, field: str = "content") -> List[str]:
    """Load documents from a JSONL file.

    Handles three shapes: a plain ``{field: str}`` record, a trace record with a
    nested/encoded ``transformed_request``/``raw_request``, or a raw line.
    """
    docs: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or len(docs) >= limit:
                continue
            record = json.loads(line)

            # Plain document record.
            if isinstance(record.get(field), str) and record[field]:
                docs.append(record[field])
                continue

            # Trace record: pull every string message content.
            for key in ("transformed_request", "raw_request", "request", "body", "payload"):
                value = record.get(key)
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except ValueError:
                        continue
                if isinstance(value, dict) and isinstance(value.get("messages"), list):
                    for message in value["messages"]:
                        content = message.get("content")
                        if isinstance(content, str) and content and len(docs) < limit:
                            docs.append(content)
                    break
    return docs[:limit]


def extract_spans(tagged: str, original: str) -> List[Tuple[str, str]]:
    """Return the ``(tag, text)`` spans the model tagged in ``tagged``.

    ``parse_tagged_string`` returns annotations against the cleaned text; since we
    compare two model outputs on the *same* original document, comparing the tag
    and the annotated value is the meaningful signal.
    """
    _, annotations = parse_tagged_string(tagged)
    return sorted((tag, text) for _, _, tag, text in annotations)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL file of documents (or a trace file)")
    ap.add_argument("--limit", type=int, default=20, help="Max documents (default 20)")
    ap.add_argument("--field", default="content", help="Document field for plain records")
    ap.add_argument("--baseline-name", default="PII-Redact-Name")
    ap.add_argument("--baseline-general", default="PII-Redact-General")
    ap.add_argument("--candidate-name", required=True)
    ap.add_argument("--candidate-general", required=True)
    ap.add_argument("--omlx-url", default="http://localhost:27473/v1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--auto-max-tokens", action="store_true", default=True)
    ap.add_argument("--mode", default="tag", choices=["tag", "redact"])
    ap.add_argument("--out", default=None, help="Write a markdown report here")
    args = ap.parse_args()

    mode = PIIHandlingMode.TAG if args.mode == "tag" else PIIHandlingMode.REDACT

    docs = load_documents(args.input, args.limit, args.field)
    if not docs:
        print("No documents loaded.", file=sys.stderr)
        return 1
    print(f"Loaded {len(docs)} documents")

    def run(name_model: str, general_model: str, label: str) -> List[str]:
        print(f"Running {label}: {name_model} + {general_model}")
        redactor = PIIRedactor(
            backend="omlx",
            omlx_base_url=args.omlx_url,
            omlx_models=[name_model, general_model],
            concurrency=args.concurrency,
            auto_max_tokens=args.auto_max_tokens,
        )
        return redactor.tag_pii_in_documents(docs, mode=mode, progress=True)

    baseline = run(args.baseline_name, args.baseline_general, "baseline")
    candidate = run(args.candidate_name, args.candidate_general, "candidate")

    # --- metrics ---------------------------------------------------------------
    n = len(docs)
    exact = 0
    tp = fp = fn = 0
    jaccard_sum = 0.0
    type_agree = 0
    char_delta = 0

    per_doc = []
    for i, (base_out, cand_out) in enumerate(zip(baseline, candidate)):
        base_spans = extract_spans(base_out, docs[i])
        cand_spans = extract_spans(cand_out, docs[i])
        bset, cset = Counter(base_spans), Counter(cand_spans)

        inter = sum((bset & cset).values())
        union = sum((bset | cset).values())
        tp += inter
        fn += sum(bset.values()) - inter
        fp += sum(cset.values()) - inter
        jaccard_sum += (inter / union) if union else 1.0

        if base_out == cand_out:
            exact += 1
        if Counter(t for t, _ in base_spans) == Counter(t for t, _ in cand_spans):
            type_agree += 1
        char_delta += abs(len(base_out) - len(cand_out))

        per_doc.append({
            "index": i,
            "identical": base_out == cand_out,
            "baseline_spans": len(base_spans),
            "candidate_spans": len(cand_spans),
            "span_delta": len(cand_spans) - len(base_spans),
        })

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 1.0

    report = {
        "documents": n,
        "mode": args.mode,
        "baseline": {"name": args.baseline_name, "general": args.baseline_general},
        "candidate": {"name": args.candidate_name, "general": args.candidate_general},
        "exact_match_rate": round(exact / n, 4),
        "span_precision": round(precision, 4),
        "span_recall": round(recall, 4),
        "span_f1": round(f1, 4),
        "span_jaccard_mean": round(jaccard_sum / n, 4),
        "tag_type_agreement": round(type_agree / n, 4),
        "mean_abs_char_delta": round(char_delta / n, 1),
        "spans_baseline": tp + fn,
        "spans_candidate": tp + fp,
        "per_doc": per_doc,
    }

    print(json.dumps({k: v for k, v in report.items() if k != "per_doc"}, indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(f"# Quant A/B: {args.candidate_name} vs {args.baseline_name}\n\n")
            fh.write(f"- documents: {n}\n- mode: {args.mode}\n")
            fh.write(f"- baseline: `{args.baseline_name}` + `{args.baseline_general}`\n")
            fh.write(f"- candidate: `{args.candidate_name}` + `{args.candidate_general}`\n\n")
            fh.write("| metric | value |\n|---|---|\n")
            for key in ("exact_match_rate", "span_precision", "span_recall", "span_f1",
                        "span_jaccard_mean", "tag_type_agreement", "mean_abs_char_delta",
                        "spans_baseline", "spans_candidate"):
                fh.write(f"| {key} | {report[key]} |\n")
            fh.write("\n## Per-document\n\n| # | identical | baseline spans | candidate spans | delta |\n")
            fh.write("|---|---|---|---|---|\n")
            for row in per_doc:
                fh.write(f"| {row['index']} | {row['identical']} | {row['baseline_spans']} | "
                         f"{row['candidate_spans']} | {row['span_delta']:+d} |\n")
        print(f"\nReport written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
