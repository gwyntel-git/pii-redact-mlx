#!/usr/bin/env python
"""Regression test for the OMLX_TIMEOUT override (see CHANGES-vs-upstream.md §3b).

Runs with plain python3 -- no pytest required:

    python3 tests/test_omlx_timeout.py

Why this test exists
--------------------
The timeout was originally read in a module-level assignment:

    DEFAULT_OMLX_TIMEOUT = float(os.environ.get("OMLX_TIMEOUT", "300"))

That freezes the value on FIRST IMPORT, so the override silently depends on
import ORDER.  If anything imports ``backends`` before OMLX_TIMEOUT is set, the
variable is missed, the 300 s default returns, and a long redaction run dies
with ``Read timed out. (read timeout=300.0)`` -- the exact failure the override
exists to prevent.  The timeout is now resolved at backend construction.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pii_redaction.backends as B  # noqa: E402

FALLBACK = 300.0


def _backend(timeout=None):
    kwargs = {"base_url": "http://127.0.0.1:1/v1"}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return B.OMLXBackend("test-model", **kwargs)


def test_env_set_after_import_still_applies():
    """The case that used to fail silently -- env set post-import."""
    os.environ.pop("OMLX_TIMEOUT", None)
    importlib.reload(B)
    assert _backend().timeout == FALLBACK
    os.environ["OMLX_TIMEOUT"] = "1800"
    assert _backend().timeout == 1800.0


def test_env_set_before_import_applies():
    os.environ["OMLX_TIMEOUT"] = "1800"
    importlib.reload(B)
    assert _backend().timeout == 1800.0


def test_explicit_argument_beats_environment():
    os.environ["OMLX_TIMEOUT"] = "1800"
    importlib.reload(B)
    assert _backend(timeout=42).timeout == 42.0


def test_junk_values_fall_back_instead_of_raising():
    importlib.reload(B)
    for junk in ("", "   ", "abc", "0", "-5"):
        os.environ["OMLX_TIMEOUT"] = junk
        assert _backend().timeout == FALLBACK, f"junk={junk!r}"


def test_module_constant_is_a_plain_number():
    """Importers may still read the constant; it must not be env-freezing."""
    importlib.reload(B)
    assert B.DEFAULT_OMLX_TIMEOUT == FALLBACK
    assert isinstance(B.resolve_omlx_timeout(), float)


if __name__ == "__main__":
    os.environ.pop("OMLX_TIMEOUT", None)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print("\nall timeout tests passed" if not failures else f"\n{failures} FAILED")
    sys.exit(1 if failures else 0)
