"""Tests for rules.py — bullet and document validators."""
import pytest
from tailor.rules import (
    check_no_em_dashes,
    check_ascii_only,
    check_word_count,
    check_starts_with_action_verb,
    check_no_filler,
    check_no_passive_open,
    check_has_metric,
    check_no_keyword_stuffing,
    check_rules,
    check_unique_action_verbs,
    check_no_repeated_tech_in_role,
    check_total_word_count,
    check_keyword_coverage,
)

GOOD_BULLET = (
    "Engineered Kafka-based event pipeline on AWS ECS, processing 8M events/day "
    "with sub-50ms p99 latency and zero message loss across 12 consumer groups."
)


def test_em_dash_detected():
    assert check_no_em_dashes("Built service — handled auth") != []

def test_en_dash_detected():
    assert check_no_em_dashes("Reduced latency – by 40%") != []

def test_no_dash_ok():
    assert check_no_em_dashes("Built service, handled auth") == []


def test_non_ascii_detected():
    assert check_ascii_only("Built café platform") != []  # é is non-ASCII

def test_bullet_chars_allowed():
    assert check_ascii_only("Built service on AWS") == []

def test_pure_ascii_ok():
    assert check_ascii_only("Built service, processed 1M events.") == []


def test_word_count_too_short():
    assert check_word_count("Built service fast.", lo=26, hi=32) != []

def test_word_count_too_long():
    long = " ".join(["word"] * 35)
    assert check_word_count(long, lo=26, hi=32) != []

def test_word_count_in_range():
    text = " ".join(["word"] * 28)
    assert check_word_count(text, lo=26, hi=32) == []

def test_word_count_lower_bound():
    text = " ".join(["word"] * 26)
    assert check_word_count(text, lo=26, hi=32) == []

def test_word_count_upper_bound():
    text = " ".join(["word"] * 32)
    assert check_word_count(text, lo=26, hi=32) == []


def test_starts_with_action_verb_ok():
    assert check_starts_with_action_verb("Engineered microservices on AWS") == []

def test_starts_with_non_verb():
    assert check_starts_with_action_verb("The team built a service") != []


def test_filler_detected():
    assert check_no_filler("Successfully shipped the feature") != []

def test_responsible_for_detected():
    assert check_no_filler("Responsible for backend services") != []

def test_no_filler_ok():
    assert check_no_filler("Engineered backend microservices") == []


def test_passive_was_detected():
    assert check_no_passive_open("Was tasked with building the API") != []

def test_passive_assisted_detected():
    assert check_no_passive_open("Assisted the team in deploying") != []

def test_active_ok():
    assert check_no_passive_open("Built the API with Python") == []

def test_starts_with_i():
    assert check_no_passive_open("I built the platform.") != []


def test_metric_percent():
    assert check_has_metric("Reduced latency 40% by caching") == []

def test_metric_multiplier():
    assert check_has_metric("Increased throughput 3x") == []

def test_metric_ms():
    assert check_has_metric("Reduced p95 latency 50ms") == []

def test_no_metric():
    assert check_has_metric("Built a microservice for data processing") != []


def test_stuffing_detected():
    stuffed = (
        "Built Python Flask Django FastAPI Redis PostgreSQL Kafka "
        "Kubernetes Docker AWS microservices service"
    )
    assert check_no_keyword_stuffing(stuffed, threshold=4) != []

def test_normal_tech_count_ok():
    normal = "Engineered Python FastAPI microservice on AWS, reducing latency 40%."
    assert check_no_keyword_stuffing(normal, threshold=4) == []


def test_good_bullet_minimal_issues():
    issues = check_rules(GOOD_BULLET)
    non_wc = [i for i in issues if "word count" not in i]
    assert non_wc == [], f"Unexpected issues: {non_wc}"


def test_unique_action_verbs_dupe():
    bullets = ["Built service A with Python", "Built service B with Java", "Deployed cluster"]
    assert check_unique_action_verbs(bullets) != []

def test_unique_action_verbs_ok():
    bullets = ["Built service A", "Deployed cluster", "Engineered pipeline"]
    assert check_unique_action_verbs(bullets) == []

def test_no_repeated_tech_in_role_dupe():
    bullets = ["Built Python API with Redis caching", "Deployed Python service on AWS"]
    assert check_no_repeated_tech_in_role(bullets) != []

def test_no_repeated_tech_in_role_ok():
    bullets = ["Built Python API on AWS", "Deployed Kubernetes cluster with Terraform"]
    assert check_no_repeated_tech_in_role(bullets) == []

def test_total_word_count_in_range():
    text = " ".join(["word"] * 500)
    assert check_total_word_count(text, lo=450, hi=600) == []

def test_total_word_count_out_of_range():
    text = " ".join(["word"] * 300)
    assert check_total_word_count(text, lo=450, hi=600) != []

def test_keyword_coverage_missing():
    tiered = {"CRITICAL": ["Kubernetes"], "IMPORTANT": []}
    bullets = ["Built Python service on AWS"]
    assert check_keyword_coverage(bullets, tiered) != []

def test_keyword_coverage_ok():
    tiered = {"CRITICAL": ["Kubernetes"], "IMPORTANT": []}
    bullets = ["Deployed service on Kubernetes with Docker"]
    assert check_keyword_coverage(bullets, tiered) == []
