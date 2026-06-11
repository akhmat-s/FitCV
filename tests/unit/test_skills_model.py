"""Unit tests for the category-based Skills model.

The Skills section drops the fixed `Relevant / Hard / Soft` buckets in favour of
JD-derived domain categories: `Skills.categories: list[Category]`, where each
`Category` is `{category: str, keywords: list[str]}`. These guard the new shape and
assert the old fixed buckets are gone (no silent fallback).
"""

from __future__ import annotations

from helprers.cv_template import Category, Skills


def test_skills_is_a_list_of_categories() -> None:
    skills = Skills(categories=[Category(category="Languages", keywords=["Python", "Go"])])
    assert skills.categories[0].category == "Languages"
    assert skills.categories[0].keywords == ["Python", "Go"]


def test_skills_defaults_to_empty_categories() -> None:
    assert Skills().categories == []


def test_category_defaults_to_empty_keywords() -> None:
    assert Category(category="Tools").keywords == []


def test_old_fixed_buckets_are_removed() -> None:
    skills = Skills(categories=[Category(category="Tools", keywords=["Docker"])])
    assert not hasattr(skills, "relevant")
    assert not hasattr(skills, "hard_skills")
    assert not hasattr(skills, "soft_skills")
