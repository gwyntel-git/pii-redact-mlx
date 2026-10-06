#!/usr/bin/env python3
"""Regression: parse_tagged_string must not raise on a tag with no closer.

An echo model that stops mid-tag emits an unterminated <PII:...>.  The parser
did `i = m.end()` then indexed `tagged_str[i]`, so a tag at end-of-string raised
IndexError and killed the run (observed in production: it cost a 4h Mac session
after ~48 parts, and would have killed a 10.5h Kaggle session the same way).

Run with: python3 tests/test_parse_tagged_string.py   (plain python3, no pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pii_redaction.redactor import parse_tagged_string     # noqa: E402

CASES = [
    # (input, expected_clean) -- expected_clean None means "just must not raise"
    ("hello world <PII:name>", "hello world "),
    ("<PII:email>", ""),
    ("a <PII:x>b</PII:x> c <PII:y>", "a b c "),
    ("hello <PII:name>world", "hello world"),
    ("hello <PII:name>Sarah</PII:name> there", "hello Sarah there"),
    ("", ""),
    ("no tags at all", "no tags at all"),
    ("<PII:a></PII:a>", ""),
    ("x<PII:", "x<PII:"),                 # no closing ">": not a tag, copied as-is
]

failed = []
for text, expected in CASES:
    try:
        clean, annots = parse_tagged_string(text)
    except Exception as e:
        failed.append("%r raised %s: %s" % (text, type(e).__name__, e))
        continue
    if expected is not None and clean != expected:
        failed.append("%r -> %r (expected %r)" % (text, clean, expected))

if failed:
    print("FAIL (%d/%d)" % (len(failed), len(CASES)))
    for f in failed:
        print("  -", f)
    sys.exit(1)

print("PASS (%d/%d cases)" % (len(CASES), len(CASES)))
