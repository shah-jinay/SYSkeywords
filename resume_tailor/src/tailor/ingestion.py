"""Parse plain-text resumes into structured Bullet records and full ResumeStructure.

Supported formats
-----------------
master_resume.txt   role on one line (date aligned right), company next line,
                    bullet char: U+00B7 (·)
role_resumes/*.txt  "Role - Company | Location | Date" on one line,
                    bullet char: U+2022 (•) or U+00B7 (·)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

BULLET_CHARS: frozenset[str] = frozenset({"·", "•"})  # U+00B7, U+2022

_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\s+\d{4}"
    r"|\b\d{2}/\d{4}"
    r"|\bPresent\b",
    re.IGNORECASE,
)

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

_SECTION_RE = re.compile(r"^[A-Z][A-Z\s]+$")
_SKILLS_CAT_RE = re.compile(r"^([^:]+):\s*(.+)$")

_KNOWN_SECTIONS: frozenset[str] = frozenset({
    "SUMMARY", "PROFESSIONAL SUMMARY",
    "TECHNICAL SKILLS", "SKILLS",
    "EXPERIENCE", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE",
    "PROJECTS", "PERSONAL PROJECTS",
    "EDUCATION", "CERTIFICATIONS",
})

_KNOWN_COMPANIES: frozenset[str] = frozenset({
    "ASU Decision Theater",
    "KiVee Softech",
    "Kintu Designs",
})

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            _nlp.add_pipe("sentencizer")
        except Exception:
            _nlp = False  # sentinel so we don't retry
    return _nlp if _nlp is not False else None


# ---------------------------------------------------------------------------
# Bullet dataclass
# ---------------------------------------------------------------------------

@dataclass
class Bullet:
    text: str
    section: str
    company: str
    role: str
    source_file: str
    action_verb: str = field(default="", init=True)
    has_metric: bool = field(default=False, init=True)
    word_count: int = field(default=0, init=True)

    def __post_init__(self):
        self.word_count = len(self.text.split())
        if not self.has_metric:
            self.has_metric = bool(_METRIC_RE.search(self.text))
        if not self.action_verb:
            self.action_verb = _extract_action_verb(self.text)


def _extract_action_verb(text: str) -> str:
    """Return first action verb from bullet text (spaCy or fallback)."""
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text[:200])
        for token in doc:
            if token.i == 0 and token.pos_ in ("VERB", "PROPN"):
                return token.lemma_.capitalize()
    first = text.split()[0] if text else ""
    return first if first and first[0].isupper() else ""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _is_section_header(line: str) -> bool:
    s = line.strip()
    return bool(_SECTION_RE.match(s)) and 3 <= len(s) <= 35 and s in _KNOWN_SECTIONS


def _starts_with_bullet(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and stripped[0] in BULLET_CHARS


def _strip_bullet(line: str) -> str:
    s = line.strip()
    while s and s[0] in BULLET_CHARS:
        s = s[1:]
    return s.strip()


def _first_part(line: str) -> str:
    """Extract content before 3+ consecutive spaces (column-aligned lines)."""
    return re.split(r"\s{3,}", line.strip())[0].strip()


def _parse_inline_header(line: str) -> tuple[str, str]:
    """
    Parse 'Role - Company | Location | Date'.
    Returns (role, company).
    """
    entry = line.split("|")[0].strip()
    if " - " in entry:
        a, b = entry.split(" - ", 1)
        a, b = a.strip(), b.strip()
        if any(c.lower() in a.lower() for c in _KNOWN_COMPANIES):
            return b, a
        return a, b
    return entry, ""


# ---------------------------------------------------------------------------
# Unified line parser
# ---------------------------------------------------------------------------

def _parse_lines(lines: list[str], source_file: str) -> list[Bullet]:
    bullets: list[Bullet] = []
    current_section = ""
    current_role = ""
    current_company = ""
    awaiting_company = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if _is_section_header(stripped):
            current_section = stripped
            current_role = ""
            current_company = ""
            awaiting_company = False
            continue

        if _starts_with_bullet(line):
            text = _strip_bullet(line)
            if text and len(text.split()) >= 4:
                bullets.append(Bullet(
                    text=text,
                    section=current_section,
                    company=current_company,
                    role=current_role,
                    source_file=source_file,
                ))
            continue

        if current_section in ("PROJECTS", "PERSONAL PROJECTS") and " | " in stripped:
            current_company = stripped.split(" | ")[0].strip()
            current_role = "Projects"
            awaiting_company = False
            continue

        if (
            current_section in ("EXPERIENCE", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE")
            and _DATE_RE.search(line)
        ):
            if "|" in line:
                role_text, company_text = _parse_inline_header(line)
                current_role = role_text
                current_company = company_text
                awaiting_company = False
            else:
                role_text = _first_part(line)
                if role_text and not _starts_with_bullet(line):
                    current_role = role_text
                    current_company = ""
                    awaiting_company = True
            continue

        if awaiting_company and current_section in (
            "EXPERIENCE", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE"
        ):
            company_text = _first_part(stripped)
            if company_text and not _starts_with_bullet(line):
                current_company = company_text
            awaiting_company = False
            continue

    return bullets


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------

def parse_master(resume_path: Path) -> list[Bullet]:
    """Parse master_resume.txt."""
    return _parse_lines(resume_path.read_text(encoding="utf-8").splitlines(), source_file="master")


def parse_role_resume(resume_path: Path) -> list[Bullet]:
    """Parse a role resume file."""
    return _parse_lines(
        resume_path.read_text(encoding="utf-8").splitlines(),
        source_file=resume_path.stem,
    )


def parse_bullets(resume_path: Path, source_file: str = "") -> list[dict]:
    """Backward-compatible dict-returning parser (used by old tests)."""
    src = source_file or resume_path.stem
    raw = _parse_lines(resume_path.read_text(encoding="utf-8").splitlines(), source_file=src)
    return [
        {
            "text": b.text,
            "section": b.section,
            "company": b.company,
            "role": b.role,
            "source_file": b.source_file,
        }
        for b in raw
    ]


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

def ingest_resume(resume_path: Path, source_file: str = "") -> int:
    """Parse one file and upsert all bullets. Returns count."""
    from .library import upsert_bullet, init_dbs
    init_dbs()
    stem = source_file or resume_path.stem
    bullets = _parse_lines(resume_path.read_text(encoding="utf-8").splitlines(), source_file=stem)
    for b in bullets:
        upsert_bullet(b)
    return len(bullets)


def ingest_all(master_resume: Path, role_resumes_dir: Path) -> dict[str, int]:
    """
    Truncate library then re-ingest master + all role resumes.
    Returns {source_stem: bullet_count}.
    """
    from .library import clear_library, init_dbs
    init_dbs()
    clear_library()

    counts: dict[str, int] = {}
    if master_resume.exists():
        counts["master"] = _ingest_file(master_resume, "master")
    if role_resumes_dir.exists():
        for txt in sorted(role_resumes_dir.glob("*.txt")):
            counts[txt.stem] = _ingest_file(txt, txt.stem)
    return counts


def _ingest_file(path: Path, source: str) -> int:
    from .library import upsert_bullet
    bullets = _parse_lines(path.read_text(encoding="utf-8").splitlines(), source_file=source)
    for b in bullets:
        upsert_bullet(b)
    return len(bullets)


# ---------------------------------------------------------------------------
# Resume structure dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResumeHeader:
    name: str
    title: str
    phone: str
    email: str
    linkedin: str
    github: str


@dataclass
class ResumeExp:
    role: str
    company: str
    dates: str
    location: str
    bullets: list[str] = field(default_factory=list)


@dataclass
class ResumeEdu:
    school: str
    degree: str
    dates: str
    location: str


@dataclass
class ResumeProject:
    name: str
    tech: str
    bullets: list[str] = field(default_factory=list)


@dataclass
class ResumeStructure:
    header: ResumeHeader
    summary: list[str]
    skills: dict[str, list[str]]
    experience: list[ResumeExp]
    education: list[ResumeEdu]
    projects: list[ResumeProject] = field(default_factory=list)


def _extract_trailing(line: str) -> tuple[str, str]:
    """Split 'Main Text    Trailing Text' on 3+ spaces. Returns (main, trailing)."""
    parts = re.split(r"\s{3,}", line.rstrip())
    if len(parts) >= 2:
        return parts[0].strip(), parts[-1].strip()
    return line.strip(), ""


def parse_resume_structure(resume_path: Path) -> ResumeStructure:
    """Parse master_resume.txt into a full ResumeStructure (header, summary, skills, experience, education)."""
    lines = resume_path.read_text(encoding="utf-8-sig").splitlines()

    # --- header: non-empty lines before the first known section ---
    header_lines: list[str] = []
    section_start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s in _KNOWN_SECTIONS:
            section_start = i
            break
        if s:
            header_lines.append(s)

    name  = header_lines[0] if len(header_lines) > 0 else ""
    htitle = header_lines[1] if len(header_lines) > 1 else ""
    contact_raw = header_lines[2] if len(header_lines) > 2 else ""
    cparts = [p.strip() for p in contact_raw.split("|") if p.strip()]
    phone    = cparts[0] if len(cparts) > 0 else ""
    email    = cparts[1] if len(cparts) > 1 else ""
    linkedin = cparts[2] if len(cparts) > 2 else ""
    github   = cparts[3] if len(cparts) > 3 else ""
    rh = ResumeHeader(name=name, title=htitle, phone=phone,
                      email=email, linkedin=linkedin, github=github)

    summary: list[str] = []
    skills: dict[str, list[str]] = {}
    experience: list[ResumeExp] = []
    education: list[ResumeEdu] = []
    projects: list[ResumeProject] = []

    current_section = ""
    current_exp: ResumeExp | None = None
    current_proj: ResumeProject | None = None
    awaiting_company = False
    edu_school = ""
    edu_dates = ""
    awaiting_edu_degree = False

    for line in lines[section_start:]:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped in _KNOWN_SECTIONS:
            if current_section in ("EXPERIENCE", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE"):
                if current_exp:
                    experience.append(current_exp)
                    current_exp = None
            current_section = stripped
            awaiting_company = False
            awaiting_edu_degree = False
            continue

        if current_section in ("SUMMARY", "PROFESSIONAL SUMMARY"):
            if _starts_with_bullet(line):
                t = _strip_bullet(line)
                if t:
                    summary.append(t)
            continue

        if current_section in ("TECHNICAL SKILLS", "SKILLS"):
            m = _SKILLS_CAT_RE.match(stripped)
            if m:
                cat = m.group(1).strip()
                raw = re.sub(r"\([^)]*\)", "", m.group(2))
                items = [x.strip() for x in raw.split(",") if x.strip()]
                skills[cat] = items
            continue

        if current_section in ("EXPERIENCE", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE"):
            if _starts_with_bullet(line):
                t = _strip_bullet(line)
                if t and len(t.split()) >= 4 and current_exp is not None:
                    current_exp.bullets.append(t)
                awaiting_company = False
                continue

            if _DATE_RE.search(line) and not _is_section_header(stripped):
                main, dates = _extract_trailing(line)
                if main:
                    if current_exp:
                        experience.append(current_exp)
                    current_exp = ResumeExp(role=main, company="", dates=dates, location="")
                    awaiting_company = True
                continue

            if awaiting_company:
                if not _starts_with_bullet(line) and not _is_section_header(stripped):
                    main, loc = _extract_trailing(line)
                    if main and current_exp:
                        current_exp.company = main
                        current_exp.location = loc
                awaiting_company = False
                continue

            # subsection labels (e.g. "Frontend & Web") — skip
            continue

        if current_section in ("PROJECTS", "PERSONAL PROJECTS"):
            if " | " in stripped and not _starts_with_bullet(line):
                # "Name | tech, stack, ..." — start a new project
                parts = stripped.split(" | ", 1)
                current_proj = ResumeProject(
                    name=parts[0].strip(),
                    tech=parts[1].strip() if len(parts) > 1 else "",
                )
                projects.append(current_proj)
            elif _starts_with_bullet(line):
                t = _strip_bullet(line)
                if t and len(t.split()) >= 4 and current_proj is not None:
                    current_proj.bullets.append(t)
            # subsection labels ("What it is", "Core Pipeline", etc.) — skip
            continue

        if current_section == "EDUCATION":
            if awaiting_edu_degree:
                main, loc = _extract_trailing(line)
                education.append(ResumeEdu(
                    school=edu_school, degree=main,
                    dates=edu_dates, location=loc,
                ))
                awaiting_edu_degree = False
            elif _DATE_RE.search(line):
                edu_school, edu_dates = _extract_trailing(line)
                awaiting_edu_degree = True
            continue

    if current_exp:
        experience.append(current_exp)

    return ResumeStructure(
        header=rh,
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
    )
