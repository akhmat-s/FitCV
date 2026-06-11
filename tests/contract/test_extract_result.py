"""Contract test: the `ExtractResult` envelope shape.

The internal envelope is
``{ "facts": {...CVFacts...}, "jd": {...JdAnalysis...}, "flags": ["string"] }``.
"""

from __future__ import annotations

from schemas import CVFacts, ExtractResult, JdAnalysis


def _extract_result() -> ExtractResult:
    facts = CVFacts.model_validate(
        {
            "personal_info": {
                "name": "Ada Lovelace",
                "location": "London, UK",
                "email": "ada@example.com",
            }
        }
    )
    jd = JdAnalysis.model_validate(
        {
            "role_title": "Engineer",
            "company": "Globex",
            "keywords": ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"],
            "candidate_level": "senior_ic",
        }
    )
    return ExtractResult(facts=facts, jd=jd, flags=["keyword-gap"])


def test_extract_result_envelope_top_level_keys() -> None:
    payload = _extract_result().model_dump()

    assert set(payload.keys()) == {"facts", "jd", "flags"}


def test_extract_result_envelope_member_types() -> None:
    payload = _extract_result().model_dump()

    assert isinstance(payload["facts"], dict)
    assert isinstance(payload["jd"], dict)
    assert isinstance(payload["flags"], list)
    assert all(isinstance(flag, str) for flag in payload["flags"])


def test_extract_result_facts_and_jd_carry_core_fields() -> None:
    payload = _extract_result().model_dump()

    # facts surfaces the parsed personal_info; jd surfaces the analysis core fields.
    assert payload["facts"]["personal_info"]["name"] == "Ada Lovelace"
    assert payload["jd"]["role_title"] == "Engineer"
    assert payload["jd"]["keywords"]
