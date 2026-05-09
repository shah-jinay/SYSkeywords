"""Render a TailoredDoc into a professional one-page PDF resume."""
from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime

# ── Layout constants ──────────────────────────────────────────────────────────
_LM   = 14.0   # left / right margin (mm)
_TM   = 11.0   # top margin
_BM   = 9.0    # bottom margin
_BH   = 4.3    # body line height (mm) — tighter to fit one page
_SW   = 36.0   # skills label column width (mm)

_MAX_SKILL_CATS     = 5   # categories shown
_MAX_SKILL_ITEMS    = 6   # items per category
_MAX_BULLETS        = 3   # bullets per experience entry
_MAX_PROJECTS       = 2   # projects shown
_MAX_PROJ_BULLETS   = 1   # bullets per project (1 tight overview line)
_BULLET_CHAR        = "\xb7"   # middle dot ·  (Latin-1 0xB7)

_UNICODE_MAP = str.maketrans({
    "–": "-",   # en dash
    "—": "--",  # em dash
    "‘": "'",   # left '
    "’": "'",   # right '
    "“": '"',   # left "
    "”": '"',   # right "
    "•": "-",   # bullet •
    "·": "-",   # middle dot ·
})


def _ascii(text: str) -> str:
    return text.translate(_UNICODE_MAP).encode("ascii", errors="ignore").decode("ascii")


def _rule(pdf, lw: float = 0.3) -> None:
    y = pdf.get_y()
    pdf.set_line_width(lw)
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)


def _sec(pdf, title: str) -> None:
    """Compact section header: ALL-CAPS bold label + thin rule + gap."""
    from fpdf import XPos, YPos
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(0, 5.5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _rule(pdf, lw=0.35)
    pdf.ln(1.5)


def _row2(pdf, left: str, right: str,
          lfont: str = "B", lsz: float = 9.5,
          rfont: str = "",  rsz: float = 8.5,
          h: float = 5.0) -> None:
    """One line: left-aligned text + right-aligned text."""
    from fpdf import XPos, YPos
    lw = pdf.epw * 0.68
    rw = pdf.epw * 0.32
    pdf.set_font("Helvetica", lfont, lsz)
    pdf.cell(lw, h, _ascii(left),  new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", rfont, rsz)
    pdf.cell(rw, h, _ascii(right), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf(doc_or_draft, out_path: Path = None) -> Path:
    from fpdf import FPDF, XPos, YPos
    from .config import settings

    settings.out_dir.mkdir(parents=True, exist_ok=True)

    is_dict = isinstance(doc_or_draft, dict)
    slug = doc_or_draft.get("jd_slug", "web") if is_dict else getattr(doc_or_draft, "jd_slug", "web")
    role = doc_or_draft.get("role", "resume") if is_dict else getattr(doc_or_draft, "role", "resume")

    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = settings.out_dir / f"{slug}_{role or 'resume'}_{ts}.pdf"

    has_structure = (
        not is_dict
        and bool(getattr(doc_or_draft, "header", None))
        and bool(getattr(doc_or_draft, "experience", None))
    )
    if not has_structure:
        return _legacy_pdf(doc_or_draft, out_path)

    doc = doc_or_draft
    pdf = FPDF(unit="mm", format="Letter")
    pdf.set_margins(left=_LM, top=_TM, right=_LM)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=_BM)

    # ── HEADER ───────────────────────────────────────────────────────────────
    h       = doc.header or {}
    name    = _ascii(h.get("name",    ""))
    htitle  = _ascii(h.get("title",   ""))
    phone   = _ascii(h.get("phone",   ""))
    email   = _ascii(h.get("email",   ""))
    linkedin= _ascii(h.get("linkedin",""))
    github  = _ascii(h.get("github",  ""))

    if name:
        pdf.set_font("Helvetica", "B", 17)
        pdf.cell(0, 8, name, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    if htitle:
        pdf.set_font("Helvetica", "", 10.5)
        pdf.cell(0, 4.5, htitle, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    contact = "  |  ".join(p for p in [phone, email, linkedin, github] if p)
    if contact:
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 4, contact, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(1.5)
    _rule(pdf, lw=0.7)   # thick rule under header
    pdf.ln(3)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    raw_summary = _ascii(getattr(doc, "summary_text", "") or "")
    if raw_summary:
        # First sentence only — keeps summary to 1-2 lines
        sents = re.split(r"(?<=[.!?])\s+", raw_summary.strip())
        summary = sents[0]
        _sec(pdf, "SUMMARY")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, _BH, summary, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1.5)

    # ── TECHNICAL SKILLS ─────────────────────────────────────────────────────
    skills = getattr(doc, "skills", {}) or {}
    if skills:
        _sec(pdf, "TECHNICAL SKILLS")
        shown = 0
        for cat, items in skills.items():
            if shown >= _MAX_SKILL_CATS:
                break
            clipped = items[:_MAX_SKILL_ITEMS]
            y = pdf.get_y()
            pdf.set_xy(pdf.l_margin, y)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(_SW, _BH, _ascii(cat) + ":")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_xy(pdf.l_margin + _SW, y)
            pdf.multi_cell(pdf.epw - _SW, _BH,
                           ", ".join(_ascii(i) for i in clipped),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            shown += 1
        pdf.ln(2)

    # ── EXPERIENCE ───────────────────────────────────────────────────────────
    experience = getattr(doc, "experience", []) or []
    if experience:
        _sec(pdf, "EXPERIENCE")
        for exp in experience:
            bullets = exp.get("bullets", [])
            if not bullets:
                continue
            # Role (bold) + dates
            _row2(pdf, exp.get("role", ""),    exp.get("dates", ""),
                  lfont="B", lsz=9.5, rfont="", rsz=8.5, h=5.0)
            # Company (italic) + location
            _row2(pdf, exp.get("company", ""), exp.get("location", ""),
                  lfont="I", lsz=9.0, rfont="", rsz=8.5, h=4.5)
            pdf.ln(0.5)
            pdf.set_font("Helvetica", "", 9)
            for b in bullets[:_MAX_BULLETS]:
                pdf.multi_cell(0, _BH, "  " + _BULLET_CHAR + "  " + _ascii(b),
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2.0)

    # ── PROJECTS ─────────────────────────────────────────────────────────────
    projects = getattr(doc, "projects", []) or []
    visible_projects = [p for p in projects if p.get("bullets")][:_MAX_PROJECTS]
    if visible_projects:
        _sec(pdf, "PROJECTS")
        for proj in visible_projects:
            name     = _ascii(proj.get("name", ""))
            tech_raw = proj.get("tech", "")
            # Truncate tech stack to first 5 items so it fits on one line
            tech_items = [t.strip() for t in tech_raw.split(",")][:5]
            tech_str = _ascii(", ".join(tech_items))
            _row2(pdf, name, tech_str, lfont="B", lsz=9.5, rfont="I", rsz=8.5, h=5.0)
            pdf.set_font("Helvetica", "", 9)
            for b in proj["bullets"][:_MAX_PROJ_BULLETS]:
                pdf.multi_cell(0, _BH, "  " + _BULLET_CHAR + "  " + _ascii(b),
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.5)

    # ── EDUCATION ────────────────────────────────────────────────────────────
    education = getattr(doc, "education", []) or []
    if education:
        _sec(pdf, "EDUCATION")
        for edu in education:
            _row2(pdf, edu.get("school", ""), edu.get("dates", ""),
                  lfont="B", lsz=9.5, rfont="", rsz=8.5, h=5.0)
            _row2(pdf, edu.get("degree", ""), edu.get("location", ""),
                  lfont="",  lsz=9.0, rfont="", rsz=8.5, h=4.5)
            pdf.ln(2)

    pdf.output(str(out_path))
    return out_path


# ── Legacy fallback ───────────────────────────────────────────────────────────

def _legacy_pdf(doc_or_draft, out_path: Path) -> Path:
    from fpdf import FPDF, XPos, YPos

    is_dict = isinstance(doc_or_draft, dict)
    title = doc_or_draft.get("title", "") if is_dict else getattr(doc_or_draft, "title", "")
    role  = doc_or_draft.get("role",  "") if is_dict else getattr(doc_or_draft, "role",  "")
    blist = doc_or_draft.get("bullets", []) if is_dict else getattr(doc_or_draft, "bullets", [])

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, _ascii(title or role or "Tailored Resume"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Experience Bullets", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    for b in blist:
        text = b["text"] if isinstance(b, dict) else b.text
        pdf.multi_cell(0, 6, f"  {_BULLET_CHAR}  {_ascii(text)}",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(str(out_path))
    return out_path


# ── Text / HTML renderers (CLI preview) ───────────────────────────────────────

def render_text(doc) -> str:
    lines: list[str] = []
    h = getattr(doc, "header", {}) or {}
    if h.get("name"):
        lines.append(h["name"])
    if h.get("title"):
        lines.append(h["title"])
    contact = " | ".join(v for k in ("phone", "email", "linkedin", "github")
                         if (v := h.get(k, "")))
    if contact:
        lines.append(contact)
    lines.append("")
    summary = getattr(doc, "summary_text", "")
    if summary:
        lines += ["SUMMARY", summary, ""]
    for cat, items in (getattr(doc, "skills", {}) or {}).items():
        lines.append(f"  {cat}: {', '.join(items)}")
    lines.append("")
    for exp in (getattr(doc, "experience", []) or []):
        lines.append(f"  {exp['role']}  {exp['dates']}")
        lines.append(f"  {exp['company']}  {exp['location']}")
        for b in exp.get("bullets", []):
            lines.append(f"    - {b}")
        lines.append("")
    lines.append("EDUCATION")
    for e in (getattr(doc, "education", []) or []):
        lines.append(f"  {e['school']}  {e['dates']}")
        lines.append(f"  {e['degree']}  {e['location']}")
    return "\n".join(lines)


def render_html_preview(doc) -> str:
    parts = ["<div class='tailored-preview'>"]
    h = getattr(doc, "header", {}) or {}
    parts.append(f"<h3>{h.get('name','') or getattr(doc,'title','') or 'Tailored Resume'}</h3>")
    parts.append("<ul>")
    for b in (getattr(doc, "bullets", []) or []):
        text   = b["text"]   if isinstance(b, dict) else b.text
        issues = b.get("issues", []) if isinstance(b, dict) else []
        cls    = "bullet-warn" if issues else "bullet-ok"
        parts.append(f'  <li class="{cls}">{text}</li>')
    parts.append("</ul>")
    gaps = getattr(doc, "gap_report", None)
    if gaps and gaps.get("CRITICAL"):
        parts.append(
            f'<p class="gap-warn">Uncovered CRITICAL: {", ".join(gaps["CRITICAL"])}</p>'
        )
    parts.append("</div>")
    return "\n".join(parts)
