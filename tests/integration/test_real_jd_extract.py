"""Live regression test: a real, full JD must yield real keywords + requirements.

REGRESSION (real JD → 0 keywords): a second ``additionalProperties`` object-map in the
analyze_jd tool schema made google/gemini-3.5-flash return a degenerate, all-empty tool
call (keywords/requirements zeroed) — a green-but-dead failure masked by synthetic stubs.

This test drives ``analyze_jd`` against a committed FULL JD fixture on the REAL model, so a
future change that silently zeros C2 again is caught. It is OPT-IN (network + real key +
cost): set ``RUN_LIVE_LLM=1`` with a valid ``OPENROUTER_API_KEY`` to run it; otherwise it
skips. The deterministic offline guard for the same root cause lives in
``tests/contract/test_tool_schemas.py::test_analyze_jd_tool_schema_has_at_most_one_object_map``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "jd_real_ai_qa.txt"

# Concrete terms the posting names verbatim — a non-empty analysis must surface them.
_EXPECTED_TERMS = ("python", "pytest", "selenium", "playwright", "docker")


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM") != "1",
    reason="live-LLM regression test (set RUN_LIVE_LLM=1 with a real OPENROUTER_API_KEY)",
)
async def test_real_full_jd_yields_keywords_and_requirements() -> None:
    # Load the REAL server-side key (override=True replaces the autouse conftest dummy key
    # on purpose — this is the one test that intentionally hits the live provider).
    load_dotenv(override=True)
    if not (os.getenv("OPENROUTER_API_KEY") or "").startswith("sk-or"):
        pytest.skip("no real OPENROUTER_API_KEY available")

    # Imported lazily so the module imports cleanly when the test is skipped.
    from extract import analyze_jd

    jd = await analyze_jd(_FIXTURE.read_text(encoding="utf-8"))

    assert jd.keywords, "real JD analysis returned ZERO keywords (C2 regression)"
    assert jd.requirements_must, "real JD analysis returned ZERO must-have requirements"

    found = {kw.lower() for kw in jd.keywords}
    hits = [term for term in _EXPECTED_TERMS if any(term in kw for kw in found)]
    assert len(hits) >= 3, f"expected concrete JD terms among keywords, got {sorted(found)}"
