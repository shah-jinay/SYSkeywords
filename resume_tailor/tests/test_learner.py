"""Tests for learner.py — Bayesian decision tracking."""
import pytest
from tailor.library import init_dbs, upsert_bullet
from tailor.learner import record_decision, bayesian_bonus, get_stats
from tailor.ingestion import Bullet


def _make_bullet(text: str) -> int:
    b = Bullet(text=text, section="EXPERIENCE", company="Acme", role="SWE", source_file="master")
    return upsert_bullet(b)


def test_bonus_no_decisions(_isolated_settings):
    init_dbs()
    assert bayesian_bonus(9999) == 0.0


def test_accept_increases_bonus(_isolated_settings):
    init_dbs()
    bid = _make_bullet("Engineered Python FastAPI service on AWS processing 1M requests daily.")
    record_decision(bid, "jd1", "accept")
    assert bayesian_bonus(bid) > 0.0


def test_bonus_bounded_above(_isolated_settings):
    init_dbs()
    bid = _make_bullet("Deployed Kubernetes cluster on AWS EKS scaling from 3 to 12 nodes.")
    for _ in range(20):
        record_decision(bid, "jd1", "accept")
    assert bayesian_bonus(bid) <= 0.15


def test_bonus_non_negative_after_accepts(_isolated_settings):
    init_dbs()
    bid = _make_bullet("Optimized PostgreSQL query latency 60% by redesigning indexes.")
    record_decision(bid, "jd1", "accept")
    record_decision(bid, "jd1", "accept")
    assert bayesian_bonus(bid) >= 0.0


def test_stats_empty_before_decisions(_isolated_settings):
    init_dbs()
    assert get_stats() == []


def test_stats_after_accept(_isolated_settings):
    init_dbs()
    bid = _make_bullet("Automated CI/CD pipeline with GitHub Actions cutting release cycle 80%.")
    record_decision(bid, "jd1", "accept")
    rows = get_stats()
    assert len(rows) == 1
    assert rows[0]["accepts"] == 1


def test_stats_filtered_by_jd(_isolated_settings):
    init_dbs()
    bid = _make_bullet("Built Redis caching layer reducing API latency 4x during traffic spikes.")
    record_decision(bid, "jd1", "accept")
    record_decision(bid, "jd2", "reject")
    assert len(get_stats(jd_slug="jd1")) == 1
    assert len(get_stats(jd_slug="jd2")) == 1


def test_edit_recorded_as_decision(_isolated_settings):
    init_dbs()
    bid = _make_bullet("Instrumented OpenTelemetry dashboards reducing MTTR from 4 hours to 30 min.")
    session_id = record_decision(bid, "jd1", "edit", edited_text="Edited version of bullet")
    assert session_id is not None
