"""Tests for matcher.py — scoring and gap report."""
import pytest
from pathlib import Path
from tailor.ingestion import Bullet
from tailor.library import init_dbs, upsert_bullet
from tailor.matcher import match_bullets_to_jd, gap_report, Match

SAMPLE_BULLETS = [
    Bullet("Engineered Kafka event pipeline on AWS ECS processing 8M events daily with sub-50ms p99 latency.",
           "EXPERIENCE", "Acme", "SWE", "master"),
    Bullet("Deployed Kubernetes cluster on AWS EKS scaling from 3 to 12 nodes during traffic spikes.",
           "EXPERIENCE", "Acme", "SWE", "master"),
    Bullet("Optimized PostgreSQL query latency 60% by redesigning indexes and adding Redis caching.",
           "EXPERIENCE", "Acme", "SWE", "master"),
    Bullet("Built Python FastAPI microservices handling 3M daily requests at 99.9% uptime.",
           "EXPERIENCE", "Acme", "SWE", "python"),
    Bullet("Automated CI/CD pipeline with GitHub Actions and Terraform cutting release cycle 80%.",
           "EXPERIENCE", "Acme", "SWE", "master"),
]

TIERED = {
    "CRITICAL": ["Kafka", "Kubernetes", "Python", "AWS"],
    "IMPORTANT": ["PostgreSQL", "Redis", "CI/CD", "Terraform"],
    "NICE_TO_HAVE": [],
}


def _seed(bullets=SAMPLE_BULLETS):
    init_dbs()
    for b in bullets:
        upsert_bullet(b)


def test_match_returns_results(_isolated_settings):
    _seed()
    matches = match_bullets_to_jd(TIERED, top_k=10)
    assert len(matches) > 0


def test_match_sorted_by_score(_isolated_settings):
    _seed()
    matches = match_bullets_to_jd(TIERED, top_k=10)
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_match_top_k_respected(_isolated_settings):
    _seed()
    matches = match_bullets_to_jd(TIERED, top_k=3)
    assert len(matches) <= 3


def test_match_has_required_fields(_isolated_settings):
    _seed()
    matches = match_bullets_to_jd(TIERED, top_k=5)
    for m in matches:
        assert m.bullet_id > 0
        assert isinstance(m.text, str) and m.text
        assert isinstance(m.score, float)
        assert m.tier in ("CRITICAL", "IMPORTANT", "NICE_TO_HAVE")


def test_source_hint_boost(_isolated_settings):
    _seed()
    no_hint   = match_bullets_to_jd(TIERED, top_k=10, source_hint="")
    with_hint = match_bullets_to_jd(TIERED, top_k=10, source_hint="python")
    snippet = "Built Python FastAPI microservices"
    s_no   = next((m.score for m in no_hint   if snippet in m.text), None)
    s_with = next((m.score for m in with_hint if snippet in m.text), None)
    if s_no is not None and s_with is not None:
        assert s_with >= s_no


def test_gap_report_finds_missing(_isolated_settings):
    _seed()
    tiered_gap = {
        "CRITICAL": ["Kafka", "GraphQL"],
        "IMPORTANT": ["PostgreSQL"],
        "NICE_TO_HAVE": [],
    }
    matches = match_bullets_to_jd(tiered_gap, top_k=10)
    gaps = gap_report(tiered_gap, matches)
    assert "GraphQL" in gaps["CRITICAL"]


def test_gap_report_no_missing(_isolated_settings):
    _seed()
    tiered_ok = {"CRITICAL": ["Kafka"], "IMPORTANT": [], "NICE_TO_HAVE": []}
    matches = match_bullets_to_jd(tiered_ok, top_k=10)
    gaps = gap_report(tiered_ok, matches)
    assert "Kafka" not in gaps.get("CRITICAL", [])


def test_empty_library_returns_empty(_isolated_settings):
    init_dbs()
    assert match_bullets_to_jd(TIERED, top_k=10) == []
