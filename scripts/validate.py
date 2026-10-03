#!/usr/bin/env python
"""
Validate PII redaction against a labelled test set.

Runs one model configuration in TAG mode over `testdata/pii_test_set.jsonl` and
scores the spans it emits against the ground-truth labels (exact containment,
per tag). This is a correctness check, not a model-vs-model comparison.

    python scripts/validate.py \
        --model-name PII-Redact-Name-oQ4 --model-general PII-Redact-General-oQ4 \
        --omlx-url http://localhost:27473/v1 --omlx-api-key-file ~/.omlx/api_key

Metrics: overall and per-tag precision / recall / F1 over (tag, text) spans.
A predicted span counts as correct when a ground-truth span of the same tag is
equal to it or contained in it (normalised: whitespace collapsed, case-folded),
absorbing harmless differences like a leading "Dr.".
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, ".")

from pii_redaction.redactor import PIIHandlingMode, PIIRedactor  # noqa: E402

DEFAULT_TEST_SET = "testdata/pii_test_set.jsonl"


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def spans_from_output(tagged: str) -> List[Tuple[str, str]]:
    """Extract (tag, text) from `<PII:tag>text</PII:tag>` markup, in order."""
    pattern = re.compile(r"<PII:([a-z_]+)>(.*?)</PII:\1>", re.DOTALL)
    return [(m.group(1), m.group(2)) for m in pattern.finditer(tagged)]


def spans_match(pred_text: str, gold_text: str) -> bool:
    p, g = normalise(pred_text), normalise(gold_text)
    return bool(p) and bool(g) and (p == g or g in p or p in g)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-set", default=DEFAULT_TEST_SET)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-general", required=True)
    ap.add_argument("--omlx-url", default="http://localhost:27473/v1")
    ap.add_argument("--omlx-api-key-file", default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None, help="Write a JSON report here")
    ap.add_argument("--show", action="store_true", help="Print each document's output")
    args = ap.parse_args()

    import os
    if args.omlx_api_key_file:
        os.environ["OMLX_API_KEY_FILE"] = args.omlx_api_key_file

    cases = []
    with open(args.test_set) as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    documents = [c["text"] for c in cases]
    print(f"Loaded {len(cases)} documents from {args.test_set}")

    redactor = PIIRedactor(
        backend="omlx",
        omlx_base_url=args.omlx_url,
        omlx_models=[args.model_name, args.model_general],
        concurrency=args.concurrency,
        auto_max_tokens=True,
    )
    outputs = redactor.tag_pii_in_documents(
        documents, mode=PIIHandlingMode.TAG, progress=True
    )

    per_tag = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    tp = fp = fn = 0
    per_doc = []

    for case, output in zip(cases, outputs):
        gold = [(t, x) for t, x in case.get("expected", [])]
        pred = spans_from_output(output)

        matched_pred = set()
        doc_fn = 0
        for gtag, gtext in gold:
            hit = None
            for i, (ptag, ptext) in enumerate(pred):
                if i in matched_pred or ptag != gtag:
                    continue
                if spans_match(ptext, gtext):
                    hit = i
                    break
            if hit is None:
                per_tag[gtag]["fn"] += 1
                fn += 1
                doc_fn += 1
            else:
                matched_pred.add(hit)
                per_tag[gtag]["tp"] += 1
                tp += 1

        doc_fp = 0
        for i, (ptag, _) in enumerate(pred):
            if i not in matched_pred:
                per_tag[ptag]["fp"] += 1
                fp += 1
                doc_fp += 1

        per_doc.append({
            "id": case["id"],
            "gold_spans": len(gold),
            "pred_spans": len(pred),
            "missed": doc_fn,
            "spurious": doc_fp,
            "output": output,
        })
        if args.show:
            print(f"\n[{case['id']}] {case['text']}")
            print(f"  -> {output}")

    def prf(t, p, f):
        precision = t / (t + p) if (t + p) else 1.0
        recall = t / (t + f) if (t + f) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 1.0
        return round(precision, 3), round(recall, 3), round(f1, 3)

    op, orr, of1 = prf(tp, fp, fn)
    report = {
        "test_set": args.test_set,
        "model_name": args.model_name,
        "model_general": args.model_general,
        "documents": len(cases),
        "gold_spans": tp + fn,
        "pred_spans": tp + fp,
        "overall": {"precision": op, "recall": orr, "f1": of1, "tp": tp, "fp": fp, "fn": fn},
        "per_tag": {
            tag: dict(zip(("precision", "recall", "f1"), prf(c["tp"], c["fp"], c["fn"])), **c)
            for tag, c in sorted(per_tag.items())
        },
        "per_document": per_doc,
    }

    print("\n" + "=" * 62)
    print(f"MODELS: {args.model_name} + {args.model_general}")
    print("=" * 62)
    print(f"overall   P={op:.3f}  R={orr:.3f}  F1={of1:.3f}   (tp={tp} fp={fp} fn={fn})")
    print(f"{'tag':<22}{'P':>7}{'R':>7}{'F1':>7}{'tp':>5}{'fp':>5}{'fn':>5}")
    for tag, c in report["per_tag"].items():
        print(f"{tag:<22}{c['precision']:>7.2f}{c['recall']:>7.2f}{c['f1']:>7.2f}"
              f"{c['tp']:>5}{c['fp']:>5}{c['fn']:>5}")
    print("\nPer-document misses:")
    for d in per_doc:
        flag = "" if (d["missed"] == 0 and d["spurious"] == 0) else "  <--"
        print(f"  {d['id']}  gold={d['gold_spans']} pred={d['pred_spans']} "
              f"missed={d['missed']} spurious={d['spurious']}{flag}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nJSON report -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
