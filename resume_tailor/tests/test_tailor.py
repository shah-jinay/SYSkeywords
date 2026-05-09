"""Tests for tailor.py — bullet selection and document assembly."""
import pytest
from tailor.ingestion import Bullet
from tailor.library import init_dbs, upsert_bullet
from tailor.matcher import Match
from tailor.tailor import select_bullets, assemble_doc, synthesize_bullet, TailoredDoc


def _match(text: str, score: float, tier: str = "CRITICAL",
           source: str = "master", bid: int = 1) -> Match:
    return Match(
        bullet_id=bid, text=text, company="Acme", section="EXPERIENCE",
        role="SWE", source_file=source, score=score, tier=tier, keyword="Python",
    )


MATCHES = [
    _match("Engineered Python FastAPI microservices on AWS ECS handling 3M requests daily at 99.9% uptime.", 0.9, bid=1),
    _match("Deployed Kubernetes cluster on AWS EKS scaling from 3 to 12 nodes for traffic spikes.", 0.8, bid=2),
    _match("Optimized PostgreSQL query latency 60% by redesigning indexes and adding Redis caching layer.", 0.75, bid=3, tier="IMPORTANT"),
    _match("Automated CI/CD pipeline with GitHub Actions and Terraform cutting release cycle from 3 days.", 0.7, bid=4),
    _match("Instrumented OpenTelemetry Prometheus dashboards reducing MTTR from 4 hours to 30 minutes.", 0.65, bid=5),
]

TIERED = {
    "CRITICAL": ["Python", "Kubernetes", "AWS"],
    "IMPORTANT": ["PostgreSQL", "CI/CD"],
    "NICE_TO_HAVE": [],
}


# ---------------------------------------------------------------------------
# select_bullets
# ---------------------------------------------------------------------------

def test_select_returns_up_to_target():
    assert len(select_bullets(MATCHES, target=3)) <= 3


def test_select_deduplicates_6word_prefix():
    dup1 = _match("Built scalable REST API using Python FastAPI on AWS handling 1M requests daily.", 0.9, bid=10)
    dup2 = _match("Built scalable REST API using Python Django on GCP handling analytics workloads.", 0.85, bid=11)
    selected = select_bullets([dup1, dup2], target=5)
    ids = [m.bullet_id for m in selected]
    assert 10 in ids
    assert 11 not in ids


def test_select_preserves_score_order():
    selected = select_bullets(MATCHES, target=5)
    scores = [m.score for m in selected]
    assert scores == sorted(scores, reverse=True)


def test_select_empty_input():
    assert select_bullets([], target=5) == []


# ---------------------------------------------------------------------------
# synthesize_bullet
# ---------------------------------------------------------------------------

def test_synthesize_known_keyword():
    text = synthesize_bullet("kubernetes")
    if text:
        assert len(text) > 10
        assert text[0].isupper()


def test_synthesize_unknown_keyword():
    assert synthesize_bullet("QuantumXYZTech2077") == ""


# ---------------------------------------------------------------------------
# assemble_doc
# ---------------------------------------------------------------------------

def test_assemble_returns_tailored_doc(_isolated_settings):
    init_dbs()
    upsert_bullet(Bullet(
        "Engineered Kafka pipeline on AWS processing 8M events daily with p99 latency.",
        "EXPERIENCE", "Acme", "SWE", "master"
    ))
    from tailor.matcher import match_bullets_to_jd
    matches = match_bullets_to_jd(TIERED, top_k=10)
    doc = assemble_doc(matches, TIERED, {"roles": ["Backend Engineer"], "_slug": "test"}, role="Backend")
    assert isinstance(doc, TailoredDoc)
    assert isinstance(doc.bullets, list)
    assert isinstance(doc.gap_report, dict)


def test_assemble_bullets_have_issues_field(_isolated_settings):
    init_dbs()
    upsert_bullet(Bullet(
        "Engineered Kafka pipeline on AWS processing 8M events daily with p99 latency.",
        "EXPERIENCE", "Acme", "SWE", "master"
    ))
    from tailor.matcher import match_bullets_to_jd
    matches = match_bullets_to_jd(TIERED, top_k=5)
    doc = assemble_doc(matches, TIERED, {})
    for b in doc.bullets:
        assert "text" in b
        assert "issues" in b
        assert isinstance(b["issues"], list)
