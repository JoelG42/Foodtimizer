"""The account namespace must stay identical to the historical Streamlit rule."""

from __future__ import annotations

import re

from foodtimizer.identity import namespace_for_subject


def _legacy(raw: str) -> str:
    return "u_" + re.sub(r"[^A-Za-z0-9]", "_", str(raw))[:80]


def test_namespace_matches_the_streamlit_rule():
    samples = [
        "109876543210987654321",
        "google-oauth2|abc",
        "Ada Lovelace <ada@example.com>",
        "üñîçødé",
        "a" * 200,
    ]
    for raw in samples:
        assert namespace_for_subject(raw) == _legacy(raw)


def test_namespace_caps_the_subject_not_the_prefix():
    namespace = namespace_for_subject("x" * 200)
    assert namespace.startswith("u_")
    assert len(namespace) == 82
