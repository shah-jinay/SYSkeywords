"""Bullet and document validation rules."""
from __future__ import annotations
import re

_EM_DASH_RE = re.compile(r"[—–]")

_FILLER_RE = re.compile(
    r"\b(?:successfully|responsible\s+for|in\s+order\s+to|helped\s+to"
    r"|assisted\s+in|worked\s+on|was\s+tasked\s+with|various|multiple"
    r"|several|leverage|utilize|synergy)\b",
    re.IGNORECASE,
)

_PASSIVE_OPENS = (
    "was ", "were ", "is ", "are ", "been ", "being ",
    "assisted ", "helped ", "supported ", "participated ",
)

_ACTION_VERBS: frozenset[str] = frozenset({
    "architected", "built", "designed", "developed", "engineered",
    "shipped", "launched", "deployed", "implemented", "migrated",
    "refactored", "optimized", "reduced", "scaled", "accelerated",
    "streamlined", "automated", "instrumented", "integrated",
    "consolidated", "overhauled", "led", "mentored", "coached",
    "hired", "owned", "spearheaded", "delivered", "drove", "cut",
    "increased", "generated", "established", "standardized",
    "negotiated", "trained", "provisioned", "structured", "secured",
    "hardened", "introduced", "published", "coordinated", "authored",
    "enforced", "modeled", "tuned", "executed", "configured",
    "documented", "added", "enabled",
})

_METRIC_RE = re.compile(
    r"\d+%"
    r"|\$[\d,]+[KMB]?"
    r"|\d+[KMB]\+"
    r"|\d+\.\d+[KMB]"
    r"|\d+x\b"
    r"|p\d{2}\b"
    r"|\d+\s*(?:ms|sec|min|hours?|days?|weeks?)"
    r"|\d+\+?\s*engineers?"
    r"|\d+\+?\s*(?:services?|teams?|users?|events?|requests?|API\s*calls?|records?)"
    r"|\d+\+?\s*(?:RPS|TPS|QPS)"
    r"|\d+\+",
    re.IGNORECASE,
)

_TECH_TERMS: frozenset[str] = frozenset({
    "python", "java", "go", "golang", "javascript", "typescript",
    "fastapi", "flask", "django", "spring", "react", "redis",
    "postgresql", "postgres", "mongodb", "kafka", "kubernetes",
    "docker", "terraform", "aws", "grpc", "opentelemetry",
    "prometheus", "grafana", "github actions", "jenkins",
    "microservices", "oauth", "jwt",
})


# ---------------------------------------------------------------------------
# Bullet-level validators
# ---------------------------------------------------------------------------

def check_no_em_dashes(text: str) -> list[str]:
    return ["contains em/en dash — use comma, colon, or 'to'"] if _EM_DASH_RE.search(text) else []


def check_ascii_only(text: str) -> list[str]:
    bad = {c for c in text if ord(c) > 127 and c not in "·•"}
    return [f"non-ASCII chars: {bad!r}"] if bad else []


def check_word_count(text: str, lo: int = 26, hi: int = 32) -> list[str]:
    wc = len(text.split())
    return [f"word count {wc} outside [{lo}, {hi}]"] if not (lo <= wc <= hi) else []


def check_starts_with_action_verb(text: str) -> list[str]:
    first = text.split()[0].lower().rstrip(".,;") if text else ""
    return [] if first in _ACTION_VERBS else [f"does not start with action verb (got '{first}')"]


def check_no_filler(text: str) -> list[str]:
    m = _FILLER_RE.search(text)
    return [f"filler word: '{m.group()}'"] if m else []


def check_no_passive_open(text: str) -> list[str]:
    low = text.lower().lstrip()
    if low.startswith("i "):
        return ["starts with 'I'"]
    for pat in _PASSIVE_OPENS:
        if low.startswith(pat):
            return [f"passive opening: '{pat.strip()}'"]
    return []


def check_has_metric(text: str) -> list[str]:
    return [] if _METRIC_RE.search(text) else ["no quantified metric"]


def check_no_keyword_stuffing(text: str, threshold: int = 4) -> list[str]:
    low = text.lower()
    hits = [t for t in _TECH_TERMS if re.search(r"\b" + re.escape(t) + r"\b", low)]
    return [f"keyword stuffing: {len(hits)} tech terms"] if len(hits) > threshold else []


def check_rules(text: str) -> list[str]:
    """Run all bullet-level validators. Returns combined violation list."""
    issues: list[str] = []
    for fn in (
        check_no_em_dashes,
        check_ascii_only,
        check_word_count,
        check_starts_with_action_verb,
        check_no_filler,
        check_no_passive_open,
        check_has_metric,
        check_no_keyword_stuffing,
    ):
        issues.extend(fn(text))
    return issues


# ---------------------------------------------------------------------------
# Document-level validators
# ---------------------------------------------------------------------------

def check_unique_action_verbs(bullets: list[str]) -> list[str]:
    verbs: list[str] = []
    for text in bullets:
        first = text.split()[0].lower().rstrip(".,;") if text else ""
        verbs.append(first)
    seen: set[str] = set()
    dupes: set[str] = set()
    for v in verbs:
        if v in seen:
            dupes.add(v)
        seen.add(v)
    return [f"repeated action verbs: {sorted(dupes)}"] if dupes else []


def check_no_repeated_tech_in_role(role_bullets: list[str]) -> list[str]:
    tech_count: dict[str, int] = {}
    for text in role_bullets:
        low = text.lower()
        for t in _TECH_TERMS:
            if re.search(r"\b" + re.escape(t) + r"\b", low):
                tech_count[t] = tech_count.get(t, 0) + 1
    dupes = [t for t, cnt in tech_count.items() if cnt > 1]
    return [f"tech repeated in same role: {sorted(dupes)}"] if dupes else []


def check_total_word_count(doc_text: str, lo: int = 450, hi: int = 600) -> list[str]:
    wc = len(doc_text.split())
    return [f"total word count {wc} outside [{lo}, {hi}]"] if not (lo <= wc <= hi) else []


def check_keyword_coverage(bullets: list[str], tiered: dict) -> list[str]:
    """Check every CRITICAL keyword appears at least once in the bullet set."""
    combined = " ".join(bullets).lower()
    missing = [
        kw for kw in tiered.get("CRITICAL", [])
        if not re.search(
            r"(?<![A-Za-z0-9])" + re.escape(kw.lower()) + r"(?![A-Za-z0-9])",
            combined,
        )
    ]
    return [f"CRITICAL keywords missing from bullets: {missing}"] if missing else []
