"""Tests for ingestion.py — parsing master and role resumes."""
import pytest
from pathlib import Path
from tailor.ingestion import parse_bullets, parse_master, parse_role_resume, BULLET_CHARS, Bullet

FIXTURES = Path(__file__).parent / "fixtures"
MASTER = Path(__file__).parent.parent / "data" / "master_resume.txt"
ROLE_DIR = Path(__file__).parent.parent / "data" / "role_resumes"


# ---------------------------------------------------------------------------
# Fixture-based tests (always run)
# ---------------------------------------------------------------------------

def test_fixture_master_parses():
    bullets = parse_bullets(FIXTURES / "sample_master.txt")
    assert len(bullets) >= 5


def test_fixture_bullet_keys():
    bullets = parse_bullets(FIXTURES / "sample_master.txt")
    for b in bullets:
        assert "text" in b
        assert "section" in b
        assert "company" in b
        assert "role" in b


def test_fixture_no_bullet_chars_in_text():
    bullets = parse_bullets(FIXTURES / "sample_master.txt")
    for b in bullets:
        for bc in BULLET_CHARS:
            assert bc not in b["text"], f"Bullet char {bc!r} leaked: {b['text'][:60]}"


def test_fixture_minimum_word_count():
    bullets = parse_bullets(FIXTURES / "sample_master.txt")
    for b in bullets:
        assert len(b["text"].split()) >= 4, f"Too short: {b['text']}"


def test_bullet_dataclass_has_metric():
    bullet = Bullet(
        text="Reduced p95 latency 40% by adding Redis caching layer",
        section="EXPERIENCE", company="Acme", role="SWE", source_file="master"
    )
    assert bullet.has_metric is True


def test_bullet_dataclass_no_metric():
    bullet = Bullet(
        text="Built backend microservices for the platform",
        section="EXPERIENCE", company="Acme", role="SWE", source_file="master"
    )
    assert bullet.has_metric is False


def test_bullet_dataclass_word_count():
    bullet = Bullet(
        text="Built backend services using Python and FastAPI",
        section="EXPERIENCE", company="Acme", role="SWE", source_file="master"
    )
    assert bullet.word_count == 7


# ---------------------------------------------------------------------------
# Real master resume tests (skip if not found)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MASTER.exists(), reason="master_resume.txt not found")
def test_master_resume_non_empty():
    bullets = parse_bullets(MASTER)
    assert len(bullets) > 30, f"Expected >30 bullets, got {len(bullets)}"


@pytest.mark.skipif(not MASTER.exists(), reason="master_resume.txt not found")
def test_master_bullet_keys():
    bullets = parse_bullets(MASTER)
    for b in bullets:
        assert "text" in b
        assert "section" in b
        assert "company" in b
        assert "role" in b


@pytest.mark.skipif(not MASTER.exists(), reason="master_resume.txt not found")
def test_master_no_bullet_chars():
    bullets = parse_bullets(MASTER)
    for b in bullets:
        for bc in BULLET_CHARS:
            assert bc not in b["text"], f"Bullet char leaked: {b['text'][:60]}"


@pytest.mark.skipif(not MASTER.exists(), reason="master_resume.txt not found")
def test_master_company_detected():
    bullets = parse_bullets(MASTER)
    exp = [b for b in bullets if b["section"] == "EXPERIENCE"]
    assert len(exp) > 20
    with_company = [b for b in exp if b["company"]]
    assert len(with_company) > 5


@pytest.mark.skipif(not MASTER.exists(), reason="master_resume.txt not found")
def test_asu_company_detected():
    bullets = parse_bullets(MASTER)
    asu = [b for b in bullets if "ASU" in b.get("company", "") or
           "Decision Theater" in b.get("company", "")]
    assert len(asu) > 0


# ---------------------------------------------------------------------------
# Role resume tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ROLE_DIR.exists(), reason="role_resumes dir not found")
def test_role_resumes_dir_exists():
    txts = list(ROLE_DIR.glob("*.txt"))
    assert len(txts) >= 1


@pytest.mark.skipif(not ROLE_DIR.exists(), reason="role_resumes dir not found")
def test_role_resume_parses_bullets():
    for txt in sorted(ROLE_DIR.glob("*.txt")):
        bullets = parse_bullets(txt, source_file=txt.stem)
        assert len(bullets) >= 5, f"{txt.name}: only {len(bullets)} bullets"


@pytest.mark.skipif(not ROLE_DIR.exists(), reason="role_resumes dir not found")
def test_role_resume_source_file_tagged():
    for txt in sorted(ROLE_DIR.glob("*.txt")):
        bullets = parse_bullets(txt, source_file=txt.stem)
        for b in bullets:
            assert b["source_file"] == txt.stem


@pytest.mark.skipif(not ROLE_DIR.exists(), reason="role_resumes dir not found")
def test_role_resume_no_bullet_chars():
    for txt in sorted(ROLE_DIR.glob("*.txt")):
        bullets = parse_bullets(txt, source_file=txt.stem)
        for b in bullets:
            for bc in BULLET_CHARS:
                assert bc not in b["text"], f"{txt.name}: char leaked: {b['text'][:60]}"


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MASTER.exists(), reason="master_resume.txt not found")
def test_ingest_command():
    from tailor.library import init_dbs, count_bullets
    from tailor.ingestion import ingest_resume
    init_dbs()
    n = ingest_resume(MASTER, source_file="master")
    assert n > 30
    assert count_bullets() == n


@pytest.mark.skipif(not MASTER.exists(), reason="master_resume.txt not found")
def test_ingest_idempotent():
    from tailor.library import init_dbs, count_bullets
    from tailor.ingestion import ingest_resume
    init_dbs()
    n1 = ingest_resume(MASTER, source_file="master")
    n2 = ingest_resume(MASTER, source_file="master")
    assert n1 == n2
    assert count_bullets() == n1
