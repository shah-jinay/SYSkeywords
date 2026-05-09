"""Pre-flight audit, bullet selection, rewriting, and document assembly."""
from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .matcher import Match


@dataclass
class TailoredDoc:
    role: str
    title: str
    company: str
    jd_slug: str
    header: dict = field(default_factory=dict)
    summary_text: str = ""
    skills: dict[str, list[str]] = field(default_factory=dict)
    experience: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    bullets: list[dict] = field(default_factory=list)  # flat list for compat
    gap_report: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bullet selection
# ---------------------------------------------------------------------------

def select_bullets(matches: list[Match], target: int = 8, role: str = "") -> list[Match]:
    """Pick up to `target` bullets, deduplicating by 6-word prefix."""
    seen: set[str] = set()
    selected: list[Match] = []

    def prefix6(text: str) -> str:
        return " ".join(text.lower().split()[:6])

    for m in matches:
        if len(selected) >= target:
            break
        p = prefix6(m.text)
        if p in seen:
            continue
        seen.add(p)
        selected.append(m)

    return selected


# ---------------------------------------------------------------------------
# Skills reordering
# ---------------------------------------------------------------------------

_SKIP_SKILL_CATS = frozenset({"certificates", "ai/ml"})


def _reorder_skills(skills: dict[str, list[str]], tiered: dict) -> dict[str, list[str]]:
    """Sort skill categories by JD keyword match count, return top 6 (excluding certificates/ai-ml)."""
    all_kw: set[str] = set()
    for kws in tiered.values():
        all_kw.update(k.lower() for k in kws)

    scored: list[tuple[int, str, list[str]]] = []
    for cat, items in skills.items():
        if cat.lower() in _SKIP_SKILL_CATS:
            continue
        n = sum(1 for item in items if any(kw in item.lower() for kw in all_kw))
        scored.append((n, cat, items))
    scored.sort(key=lambda x: -x[0])

    return {cat: items for _, cat, items in scored[:6]}


# ---------------------------------------------------------------------------
# Bullet synthesis from templates
# ---------------------------------------------------------------------------

def _load_templates() -> dict:
    from .config import settings
    p = Path(settings.prompts_dir) / "templates.yaml"
    if p.exists():
        try:
            import yaml
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def synthesize_bullet(keyword: str) -> str:
    """Generate a bullet for a gap keyword from templates.yaml."""
    import random
    templates = _load_templates()
    kw_key = re.sub(r"[^a-z0-9]", "", keyword.lower())
    entry = templates.get(kw_key) or templates.get(keyword.lower())
    if not entry:
        return ""
    verb    = random.choice(entry.get("verbs", ["Built"]))
    context = entry.get("context", f"{keyword} infrastructure")
    result  = random.choice(entry.get("result_range", ["improving system reliability"]))
    techs   = entry.get("tech", [keyword])[:3]
    return f"{verb} {context} using {', '.join(techs)}, {result}."


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def assemble_doc(
    matches: list[Match],
    tiered: dict,
    jd_data: dict,
    role: str = "",
) -> TailoredDoc:
    """
    Build a full TailoredDoc from scored matches + JD keyword tiers.
    Loads master resume structure for header/skills/education.
    Groups EXPERIENCE bullets by company and backfills thin entries.
    """
    from .matcher import gap_report
    from .config import settings
    from .library import get_bullets_for_company

    # --- group EXPERIENCE matches by company ---
    by_company: dict[str, list[Match]] = defaultdict(list)
    for m in matches:
        if m.section == "EXPERIENCE":
            by_company[m.company].append(m)

    # sort each company's matches by score
    for co in by_company:
        by_company[co].sort(key=lambda m: -m.score)

    # --- load master resume structure ---
    structure = None
    try:
        from .ingestion import parse_resume_structure
        master_path = Path(str(settings.master_resume))
        if master_path.exists():
            structure = parse_resume_structure(master_path)
    except Exception:
        pass

    # --- build experience list ---
    experience: list[dict] = []
    if structure and structure.experience:
        for exp in structure.experience:
            co = exp.company
            co_matches = by_company.get(co, [])

            # dedup by 6-word prefix
            selected_texts: list[str] = []
            seen_prefixes: set[str] = set()
            for m in co_matches:
                prefix = " ".join(m.text.lower().split()[:6])
                if prefix not in seen_prefixes:
                    seen_prefixes.add(prefix)
                    selected_texts.append(m.text)
                if len(selected_texts) >= 5:
                    break

            # backfill from library if fewer than 3 bullets
            if len(selected_texts) < 3:
                fallback = get_bullets_for_company(co, section="EXPERIENCE", limit=8)
                used = set(selected_texts)
                for b in fallback:
                    if b["text"] not in used and len(selected_texts) < 5:
                        selected_texts.append(b["text"])
                        used.add(b["text"])

            if selected_texts:
                experience.append({
                    "role":     exp.role,
                    "company":  exp.company,
                    "dates":    exp.dates,
                    "location": exp.location,
                    "bullets":  selected_texts,
                })

    # --- skills (reordered by JD relevance) ---
    raw_skills = structure.skills if structure else {}
    ordered_skills = _reorder_skills(raw_skills, tiered) if raw_skills else {}

    # --- summary text (first summary bullet) ---
    summary_text = (structure.summary[0] if structure and structure.summary else "")

    # --- header ---
    header: dict = {}
    if structure:
        h = structure.header
        header = {
            "name":     h.name,
            "title":    h.title,
            "phone":    h.phone,
            "email":    h.email,
            "linkedin": h.linkedin,
            "github":   h.github,
        }

    # --- projects (pass through top 3) ---
    projects: list[dict] = []
    if structure:
        for p in structure.projects[:3]:
            projects.append({
                "name":    p.name,
                "tech":    p.tech,
                "bullets": p.bullets,
            })

    # --- education ---
    education: list[dict] = []
    if structure:
        for e in structure.education:
            education.append({
                "school":   e.school,
                "degree":   e.degree,
                "dates":    e.dates,
                "location": e.location,
            })

    # --- flat bullets list for gap report + legacy compat ---
    all_exp_matches: list[Match] = []
    for co_list in by_company.values():
        all_exp_matches.extend(co_list)
    all_exp_matches.sort(key=lambda m: -m.score)
    flat_selected = select_bullets(all_exp_matches, target=12, role=role)
    gaps = gap_report(tiered, flat_selected)

    from .rules import check_rules
    bullet_dicts = [
        {
            "id":          m.bullet_id,
            "text":        m.text,
            "company":     m.company,
            "section":     m.section,
            "score":       m.score,
            "tier":        m.tier,
            "keyword":     m.keyword,
            "issues":      check_rules(m.text),
            "synthesized": False,
        }
        for m in flat_selected
    ]

    jd_roles = jd_data.get("roles", [])
    return TailoredDoc(
        role=role,
        title=jd_roles[0] if jd_roles else role,
        company="",
        jd_slug=jd_data.get("_slug", "web"),
        header=header,
        summary_text=summary_text,
        skills=ordered_skills,
        experience=experience,
        projects=projects,
        education=education,
        bullets=bullet_dicts,
        gap_report=gaps,
    )


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

def draft_resume(jd_slug: str, role: Optional[str] = None, target_bullets: int = 8) -> dict:
    """Thin wrapper kept for CLI backward compat. Returns legacy draft dict."""
    import json
    from .config import settings

    jd_path = Path(settings.data_dir) / "jd" / f"{jd_slug}.json"
    if not jd_path.exists():
        raise FileNotFoundError(f"JD file not found: {jd_path}")

    with open(jd_path, encoding="utf-8") as f:
        jd_raw = json.load(f)

    jd_text = jd_raw.get("jd_text", "")
    from extractor import extract  # type: ignore[import]
    jd_data = extract(jd_text)
    jd_data["_slug"] = jd_slug

    from .matcher import match_bullets_to_jd

    tiered = {
        "CRITICAL": jd_data.get("tech_skills", [])[:15],
        "IMPORTANT": (
            jd_data.get("inferred_domains", [])
            + jd_data.get("responsibilities", [])[:6]
            + jd_data.get("soft_skills", [])
        )[:12],
        "NICE_TO_HAVE": (
            jd_data.get("amazon_lps", []) + jd_data.get("behavioral_traits", [])
        )[:8],
    }

    matches = match_bullets_to_jd(tiered, top_k=60, source_hint=role or "")
    selected = select_bullets(matches, target=target_bullets, role=role or "")

    from .rules import check_rules
    return {
        "jd_slug": jd_slug,
        "role": role or "",
        "title": jd_data.get("roles", [""])[0] if jd_data.get("roles") else "",
        "company": "",
        "bullets": [
            {
                "id":      m.bullet_id,
                "text":    m.text,
                "company": m.company,
                "section": m.section,
                "score":   m.score,
                "tier":    m.tier,
                "keyword": m.keyword,
                "issues":  check_rules(m.text),
            }
            for m in selected
        ],
    }
