"""Score resume bullets against JD keyword tiers."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

import numpy as np

_SOURCE_BOOST = 0.08


def _wb_pattern(kw: str) -> re.Pattern:
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


@dataclass
class Match:
    bullet_id: int
    text: str
    company: str
    section: str
    role: str
    source_file: str
    score: float
    tier: str
    keyword: str
    issues: list[str] = field(default_factory=list)


def _kw_hit_rate(text: str, keywords: list[str]) -> float:
    """Fraction of keywords matching text (exact word-boundary + rapidfuzz fallback)."""
    if not keywords:
        return 0.0
    try:
        from rapidfuzz.fuzz import ratio as fratio
        use_fuzzy = True
    except ImportError:
        use_fuzzy = False

    hits = 0.0
    for kw in keywords:
        if _wb_pattern(kw).search(text):
            hits += 1.0
        elif use_fuzzy and fratio(kw.lower(), text.lower()) >= 90:
            hits += 0.5
    return min(hits / len(keywords), 1.0)


def _best_keyword(text: str, keywords: list[str]) -> str:
    for kw in keywords:
        if _wb_pattern(kw).search(text):
            return kw
    return keywords[0] if keywords else ""


def match_bullets_to_jd(
    tiered: dict,
    top_k: int = 60,
    source_hint: str = "",
) -> list[Match]:
    """
    Score every bullet in the library against CRITICAL/IMPORTANT tiers.
    Returns top_k matches sorted by descending score.

    Score = 0.5 * kw_critical + 0.2 * kw_important + 0.3 * semantic
    """
    from .library import get_all_bullets
    from .embeddings import get_or_compute_bullet_embeddings, embed, cosine

    critical: list[str] = tiered.get("CRITICAL", [])
    important: list[str] = tiered.get("IMPORTANT", [])
    all_kws = critical + important

    bullets = get_all_bullets()
    if not bullets:
        return []

    jd_query = " ".join(all_kws) if all_kws else "software engineer"
    jd_vec = embed([jd_query])[0]
    embeddings = get_or_compute_bullet_embeddings()

    matches: list[Match] = []
    for b in bullets:
        crit  = _kw_hit_rate(b["text"], critical)
        imp   = _kw_hit_rate(b["text"], important)
        b_vec = embeddings.get(b["id"])
        sem   = cosine(jd_vec, b_vec) if b_vec is not None else 0.0

        score = 0.5 * crit + 0.2 * imp + 0.3 * sem
        if source_hint and b.get("source_file", "").lower() == source_hint.lower():
            score = min(score + _SOURCE_BOOST, 1.0)

        tier = "CRITICAL" if crit > 0 else ("IMPORTANT" if imp > 0 else "NICE_TO_HAVE")
        keyword = _best_keyword(b["text"], all_kws)

        matches.append(Match(
            bullet_id=b["id"],
            text=b["text"],
            company=b.get("company", ""),
            section=b.get("section", ""),
            role=b.get("role", ""),
            source_file=b.get("source_file", ""),
            score=score,
            tier=tier,
            keyword=keyword,
        ))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:top_k]


def gap_report(tiered: dict, matches: list[Match]) -> dict[str, list[str]]:
    """Return keywords from each tier not found in any matched bullet text."""
    covered: set[str] = set()
    all_text = " ".join(m.text for m in matches)
    for tier in ("CRITICAL", "IMPORTANT"):
        for kw in tiered.get(tier, []):
            if _wb_pattern(kw).search(all_text):
                covered.add(kw.lower())
    return {
        "CRITICAL": [k for k in tiered.get("CRITICAL", []) if k.lower() not in covered],
        "IMPORTANT": [k for k in tiered.get("IMPORTANT", []) if k.lower() not in covered],
    }
