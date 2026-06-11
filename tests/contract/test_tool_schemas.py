"""Contract tests: function-calling tool schemas keep their contracted shape.

The tool schema passed to ``LLMModel.call_tool`` must keep the shape the model is
contracted against (name, required fields, nested properties), independent of how the
Pydantic models render their JSON schema.
"""

from __future__ import annotations

from extract import analyze_jd_tool_schema, cv_facts_tool_schema
from schemas import CandidateLevel, KeywordTier, TargetSection
from tool_schemas import (
    generate_experience_tool_schema,
    generate_language_tool_schema,
    generate_skills_tool_schema,
)

def test_cv_facts_tool_schema_top_level_shape() -> None:
    schema = cv_facts_tool_schema()

    assert schema["name"] == "extract_cv_facts"
    assert isinstance(schema["description"], str) and schema["description"]
    params = schema["parameters"]
    assert params["type"] == "object"
    # personal_info is the only top-level required field.
    assert params["required"] == ["personal_info"]


def test_cv_facts_tool_schema_personal_info_required_fields() -> None:
    schema = cv_facts_tool_schema()
    personal_info = schema["parameters"]["properties"]["personal_info"]

    assert personal_info["type"] == "object"
    # Identity only: name + email. `location` is an absent-able scalar fact — requiring it
    # pressures the eager model to fabricate a city; it stays a property, not required.
    assert set(personal_info["required"]) == {"name", "email"}
    props = personal_info["properties"]
    for field in ("name", "location", "email", "phone", "links"):
        assert field in props


def test_cv_facts_tool_schema_collection_properties_present() -> None:
    schema = cv_facts_tool_schema()
    props = schema["parameters"]["properties"]

    for collection in (
        "experiences",
        "education",
        "projects",
        "certificates",
        "languages",
        "skills",
    ):
        assert props[collection]["type"] == "array"


def test_cv_facts_tool_schema_skills_group_shape() -> None:
    schema = cv_facts_tool_schema()
    skills_items = schema["parameters"]["properties"]["skills"]["items"]

    assert skills_items["type"] == "object"
    # `items` is the group's identity; `category` is an absent-able label (kept as a property).
    assert set(skills_items["required"]) == {"items"}
    props = skills_items["properties"]
    assert props["category"] == {"type": "string"}
    assert props["items"] == {"type": "array", "items": {"type": "string"}}


def test_cv_facts_tool_schema_experience_required_fields() -> None:
    schema = cv_facts_tool_schema()
    experience_items = schema["parameters"]["properties"]["experiences"]["items"]

    # Identity only: role + company. start_date/end_date are absent-able facts (an ongoing
    # role omits end_date; a source may omit dates entirely) — requiring them is the
    # fabrication lever, so they stay properties, not required.
    assert set(experience_items["required"]) == {"role", "company"}


_CONTRACT_SECTION_ENUM = [
    "contact",
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
]
_CONTRACT_LEVEL_ENUM = [
    "new_grad",
    "entry",
    "mid",
    "senior_ic",
    "manager",
    "director",
]


def test_analyze_jd_tool_schema_top_level_shape() -> None:
    schema = analyze_jd_tool_schema()

    assert schema["name"] == "analyze_job_description"
    assert isinstance(schema["description"], str) and schema["description"]
    params = schema["parameters"]
    assert params["type"] == "object"
    # role_title/company are absent-able scalar facts (a JD may name neither) — dropped from
    # required. keywords is an array (empty [] satisfies it) and candidate_level is the
    # inferred enum; both stay required.
    assert set(params["required"]) == {"keywords", "candidate_level"}


def test_analyze_jd_tool_schema_candidate_level_enum() -> None:
    schema = analyze_jd_tool_schema()
    candidate_level = schema["parameters"]["properties"]["candidate_level"]

    assert candidate_level["type"] == "string"
    # Matches the CandidateLevel enum exactly.
    assert candidate_level["enum"] == _CONTRACT_LEVEL_ENUM
    assert candidate_level["enum"] == [level.value for level in CandidateLevel]


def test_analyze_jd_tool_schema_keyword_plan_section_enum() -> None:
    schema = analyze_jd_tool_schema()
    keyword_plan = schema["parameters"]["properties"]["keyword_plan"]

    assert keyword_plan["type"] == "object"
    section_values = keyword_plan["additionalProperties"]
    assert section_values["type"] == "string"
    # Matches the TargetSection enum exactly.
    assert section_values["enum"] == _CONTRACT_SECTION_ENUM
    assert section_values["enum"] == [section.value for section in TargetSection]


def test_analyze_jd_tool_schema_keyword_tiers_enum() -> None:
    schema = analyze_jd_tool_schema()
    keyword_tiers = schema["parameters"]["properties"]["keyword_tiers"]

    assert keyword_tiers["type"] == "array"
    item = keyword_tiers["items"]
    assert item["type"] == "object"
    assert item["required"] == ["keyword", "tier"]
    assert item["properties"]["keyword"]["type"] == "string"
    tier_values = item["properties"]["tier"]
    assert tier_values["type"] == "string"
    # Matches the KeywordTier enum exactly (two-tier evidence).
    assert tier_values["enum"] == ["concrete", "competency"]
    assert tier_values["enum"] == [tier.value for tier in KeywordTier]


def test_analyze_jd_tool_schema_has_at_most_one_object_map() -> None:
    """Root-cause regression guard (REGRESSION: real JD → 0 keywords).

    google/gemini-3.5-flash via OpenRouter degrades to a degenerate function call —
    scalar fields filled, every array/object empty — when the tool schema carries TWO
    open-ended ``additionalProperties`` object-maps. Pinning the analyze_jd schema to at
    most one such map keeps keyword/requirement extraction alive. This fails if a future
    change reintroduces a second object-map (the exact ca91ac4 regression).
    """
    props = analyze_jd_tool_schema()["parameters"]["properties"]
    object_maps = [
        name
        for name, spec in props.items()
        if spec.get("type") == "object" and "additionalProperties" in spec
    ]
    assert len(object_maps) <= 1, f"too many open object-maps: {object_maps}"


def test_generate_skills_per_keyword_category_is_free_string_not_enum() -> None:
    schema = generate_skills_tool_schema()
    keyword_item = schema["parameters"]["properties"]["categories"]["items"]["properties"][
        "keywords"
    ]["items"]
    props = keyword_item["properties"]
    # the per-keyword grouping field is `category`, a plain string with NO enum (was `bucket`)
    assert "bucket" not in props
    assert props["category"] == {"type": "string"}
    assert "enum" not in props["category"]
    assert set(keyword_item["required"]) == {"keyword", "tier", "category"}
    # tier keeps its enum; only the taxonomy went away
    assert props["tier"]["enum"] == [tier.value for tier in KeywordTier]


def test_closed_skill_taxonomy_symbols_are_removed() -> None:
    """No closed software taxonomy survives anywhere in the production sources (grep guard)."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    banned = ("CANONICAL_SKILL_BUCKETS", "_BUCKET_ENUM", "CANONICAL_BUCKETS", "_BARE_UMBRELLA")
    for module in ("schemas.py", "tool_schemas.py", "cv_generator.py"):
        source = (root / module).read_text(encoding="utf-8")
        for symbol in banned:
            assert symbol not in source, f"{symbol} still present in {module}"


def test_edited_schema_required_sets_are_identity_only() -> None:
    cv_facts = cv_facts_tool_schema()["parameters"]["properties"]
    assert set(cv_facts["projects"]["items"]["required"]) == {"name"}
    assert set(cv_facts["certificates"]["items"]["required"]) == {"title"}
    assert set(cv_facts["languages"]["items"]["required"]) == {"language"}

    gen_exp = generate_experience_tool_schema()["parameters"]["properties"]["experiences"]["items"]
    # `bullets` stays required (an array; its minItems:1 is out of scope) alongside identity.
    assert set(gen_exp["required"]) == {"role", "company", "bullets"}

    gen_lang = generate_language_tool_schema()["parameters"]["properties"]["languages"]["items"]
    assert set(gen_lang["required"]) == {"language"}
