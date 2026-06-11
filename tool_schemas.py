"""Hand-built function-calling tool schemas for the extract + section-generation passes.

These schemas are hand-built with a flat shape (no $defs/$ref), which is the contract the
model is invoked against. They are deliberately NOT derived from `model_json_schema()`
(keep the flat contract shape). The enum value lists ARE derived from the schemas.py
StrEnums so the tool schema and the validation models cannot drift. The two extract
schemas live below; the seven section-generation schemas follow.
"""

from __future__ import annotations

from schemas import (
    CandidateLevel,
    KeywordTier,
    TargetSection,
)

#: keyword_plan target sections — single source of truth is the TargetSection enum.
_SECTION_ENUM = [section.value for section in TargetSection]

#: inferred candidate level — single source of truth is the CandidateLevel enum.
_LEVEL_ENUM = [level.value for level in CandidateLevel]

#: keyword evidence tiers — single source of truth is the KeywordTier enum.
_TIER_ENUM = [tier.value for tier in KeywordTier]

_LINK_SCHEMA = {
    "type": "object",
    "required": ["url"],
    "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
}


def cv_facts_tool_schema() -> dict:
    """Return the `extract_cv_facts` function-calling tool schema."""
    return {
        "name": "extract_cv_facts",
        "description": (
            "Extract truthful structured facts from the candidate's CV text. Do not invent."
        ),
        "parameters": {
            "type": "object",
            "required": ["personal_info"],
            "properties": {
                "personal_info": {
                    "type": "object",
                    "required": ["name", "email"],
                    "properties": {
                        "name": {"type": "string"},
                        "location": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "links": {"type": "array", "items": _LINK_SCHEMA},
                    },
                },
                "experiences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["role", "company"],
                        "properties": {
                            "role": {"type": "string"},
                            "company": {"type": "string"},
                            "company_description": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "location": {"type": "string"},
                            "bullets": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["institution", "degree"],
                        "properties": {
                            "institution": {"type": "string"},
                            "degree": {"type": "string"},
                            "start_year": {"type": "integer"},
                            "end_year": {"type": "integer"},
                            "gpa": {"type": "string"},
                        },
                    },
                },
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "skills": {"type": "array", "items": {"type": "string"}},
                            "link": _LINK_SCHEMA,
                        },
                    },
                },
                "certificates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["title"],
                        "properties": {
                            "title": {"type": "string"},
                            "issuer": {"type": "string"},
                            "year": {"type": "integer"},
                            "link": _LINK_SCHEMA,
                        },
                    },
                },
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["language"],
                        "properties": {
                            "language": {"type": "string"},
                            "level": {"type": "string"},
                        },
                    },
                },
                "skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["items"],
                        "properties": {
                            "category": {"type": "string"},
                            "items": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        },
    }


def analyze_jd_tool_schema() -> dict:
    """Return the `analyze_job_description` function-calling tool schema."""
    return {
        "name": "analyze_job_description",
        "description": (
            "Analyze the job description into requirements, keywords, a keyword→section plan, "
            "and a per-keyword evidence tier (concrete | competency)."
        ),
        "parameters": {
            "type": "object",
            "required": ["keywords", "candidate_level"],
            "properties": {
                "role_title": {"type": "string"},
                "company": {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "requirements_must": {"type": "array", "items": {"type": "string"}},
                "requirements_nice": {"type": "array", "items": {"type": "string"}},
                "keyword_plan": {
                    "type": "object",
                    "additionalProperties": {"type": "string", "enum": _SECTION_ENUM},
                },
                # keyword_tiers is an ARRAY of {keyword, tier} objects, not a
                # second `additionalProperties` object-map. Two open object-maps (alongside
                # keyword_plan) made google/gemini-3.5-flash emit a degenerate tool call with
                # keywords/requirements empty (real-JD → 0 keywords regression). One typed
                # array keeps the per-keyword tier while leaving a single object-map.
                "keyword_tiers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["keyword", "tier"],
                        "properties": {
                            "keyword": {"type": "string"},
                            "tier": {"type": "string", "enum": _TIER_ENUM},
                        },
                    },
                },
                "candidate_level": {"type": "string", "enum": _LEVEL_ENUM},
            },
        },
    }


# --- Section generation tool schemas -----
# One hand-built flat schema per generated CV section, mirroring the extract schemas above
# (flat shape, no $defs/$ref). `personal_info` has NO schema — it is carried truthfully
# from CVFacts, never regenerated. Array-producing sections wrap their list in a single
# property keyed by the cv_template field name so the section generators read
# result["<field>"]. Bullet `action_verb` is an advisory string (no enum): out-of-enum is
# a validator warning, not a gate.

_BULLET_SCHEMA = {
    "type": "object",
    "required": ["action_verb", "description"],
    "properties": {
        "action_verb": {"type": "string"},
        "description": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "impact": {"type": "string"},
        "benefit": {"type": "string"},
    },
}


def generate_summary_tool_schema() -> dict:
    """Return the `generate_summary` tool schema (forces CVSummary)."""
    return {
        "name": "generate_summary",
        "description": (
            "Write the tailored professional summary (3–5 lines) surfacing the JD keywords "
            "mapped to the summary section. Reframe truthful facts only; never invent."
        ),
        "parameters": {
            "type": "object",
            "required": ["text", "relevant_skills"],
            "properties": {
                "text": {"type": "string"},
                "relevant_skills": {"type": "array", "items": {"type": "string"}},
            },
        },
    }


def generate_skills_tool_schema() -> dict:
    """Return the `generate_skills` tool schema (forces JD-derived categories)."""
    return {
        "name": "generate_skills",
        "description": (
            "Surface the candidate's truthful, JD-named skills. Give EACH keyword a short, "
            "conventional `category` header you DERIVE from this candidate's own field and the "
            "posting's wording (a free-text label — there is no fixed list). Keep each header to "
            "a short noun label of at most a few words, one concept per header, plain text with "
            "no markdown or punctuation tricks. Tag each keyword with its tier; a concrete "
            "keyword must appear literally in the CV, a competency keyword must carry an "
            "anchor_ref copied verbatim from the CV. Include ONLY skills the facts evidence AND "
            "the JD names; drop bare umbrella tokens; omit soft skills unless the JD requires "
            "them; never invent."
        ),
        "parameters": {
            "type": "object",
            "required": ["categories"],
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["category", "keywords"],
                        "properties": {
                            "category": {"type": "string"},
                            "keywords": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["keyword", "tier", "category"],
                                    "properties": {
                                        "keyword": {"type": "string"},
                                        "tier": {"type": "string", "enum": _TIER_ENUM},
                                        "category": {"type": "string"},
                                        "anchor_ref": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def generate_experience_tool_schema() -> dict:
    """Return the `generate_experience` tool schema (forces array[CVExperience])."""
    return {
        "name": "generate_experience",
        "description": (
            "Rewrite each work-experience entry truthfully against the JD keyword plan. Company "
            "names and dates must match the source facts exactly; never invent roles."
        ),
        "parameters": {
            "type": "object",
            "required": ["experiences"],
            "properties": {
                "experiences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "role",
                            "company",
                            "bullets",
                        ],
                        "properties": {
                            "role": {"type": "string"},
                            "company": {"type": "string"},
                            "company_description": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "location": {"type": "string"},
                            "bullets": {
                                "type": "array",
                                "minItems": 1,
                                "items": _BULLET_SCHEMA,
                            },
                        },
                    },
                },
            },
        },
    }


def generate_education_tool_schema() -> dict:
    """Return the `generate_education` tool schema (forces array[CVEducation])."""
    return {
        "name": "generate_education",
        "description": "Pass through the candidate's truthful education entries; never invent.",
        "parameters": {
            "type": "object",
            "required": ["education"],
            "properties": {
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["institution", "degree"],
                        "properties": {
                            "institution": {"type": "string"},
                            "degree": {"type": "string"},
                            "start_year": {"type": "integer"},
                            "end_year": {"type": "integer"},
                            "gpa": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


def generate_project_tool_schema() -> dict:
    """Return the `generate_project` tool schema (forces array[CVProject])."""
    return {
        "name": "generate_project",
        "description": (
            "Pass through only projects present in the source facts; never invent projects."
        ),
        "parameters": {
            "type": "object",
            "required": ["projects"],
            "properties": {
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "skills": {"type": "array", "items": {"type": "string"}},
                            "link": _LINK_SCHEMA,
                        },
                    },
                },
            },
        },
    }


def generate_certificate_tool_schema() -> dict:
    """Return the `generate_certificate` tool schema (forces array[CVCertificate])."""
    return {
        "name": "generate_certificate",
        "description": (
            "Pass through only certificates present in the source facts; never invent."
        ),
        "parameters": {
            "type": "object",
            "required": ["certificates"],
            "properties": {
                "certificates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["title"],
                        "properties": {
                            "title": {"type": "string"},
                            "issuer": {"type": "string"},
                            "year": {"type": "integer"},
                            "link": _LINK_SCHEMA,
                        },
                    },
                },
            },
        },
    }


def generate_language_tool_schema() -> dict:
    """Return the `generate_language` tool schema (forces array[CVLanguage])."""
    return {
        "name": "generate_language",
        "description": "Pass through only languages present in the source facts; never invent.",
        "parameters": {
            "type": "object",
            "required": ["languages"],
            "properties": {
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["language"],
                        "properties": {
                            "language": {"type": "string"},
                            "level": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


# --- Cover-letter generation tool schema -
# Flat hand-built schema mirroring the section schemas above (no $defs/$ref). A single
# function-calling step returns the prose `text` (required; guarded by `_require_fields`).
# The letter's full envelope (salutation → body → sign-off) is mandated in the system prompt
# and enforced deterministically by `cover_letter._is_well_structured`, not by the schema shape.


def generate_cover_letter_tool_schema() -> dict:
    """Return the `generate_cover_letter` tool schema."""
    return {
        "name": "generate_cover_letter",
        "description": (
            "Write a truthful, well-structured cover letter — a salutation, an opening, themed "
            "body paragraphs matching each posted JD requirement to specific evidence from the "
            "candidate's CV facts in prose, a closing, and a sign-off with the candidate's name. "
            "Reframe existing facts only; never invent."
        ),
        "parameters": {
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
            },
        },
    }
