"""
Debug mode: pipe a JD and resume through every stage and dump all intermediate
state to stdout (or a file).

Usage:
    python debug_pipeline.py --jd path/to/jd.txt --resume path/to/resume.txt
    python debug_pipeline.py --jd path/to/jd.txt --resume path/to/resume.txt --out debug.json
    python debug_pipeline.py --demo   # runs a built-in test case
"""
import argparse
import json
import sys
import os
import re

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))

from extractor import extract
from comparator import compare, _split_chunks, _is_exact, _is_implied
from normalizer import canonicalize, all_variants, wcontains, strip_version


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _dump_list(label: str, items: list, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{label} ({len(items)}):")
    for item in items:
        print(f"{pad}  • {item}")


# ---------------------------------------------------------------------------
# Main debug run
# ---------------------------------------------------------------------------

def debug(jd_text: str, resume_text: str, out_path: str | None = None) -> dict:
    trace: dict = {}

    # ── Stage 1: Raw text stats ─────────────────────────────────────────────
    _section("STAGE 1 — Raw input")
    print(f"  JD length:     {len(jd_text)} chars, {len(jd_text.splitlines())} lines")
    print(f"  Resume length: {len(resume_text)} chars, {len(resume_text.splitlines())} lines")
    trace["input"] = {
        "jd_chars": len(jd_text),
        "resume_chars": len(resume_text),
    }

    # ── Stage 2: JD extraction ──────────────────────────────────────────────
    _section("STAGE 2 — JD extraction")
    jd_data = extract(jd_text)

    for cat, items in jd_data.items():
        if isinstance(items, list) and items:
            _dump_list(cat, items)
    trace["jd_data"] = jd_data

    # ── Stage 3: Normalization check ────────────────────────────────────────
    _section("STAGE 3 — Normalization spot-check (tech skills)")
    for skill in jd_data.get("tech_skills", []):
        canon = canonicalize(skill)
        stripped = strip_version(skill)
        variants = sorted(all_variants(skill))
        print(f"  [{skill}] → canon={canon!r}, stripped={stripped!r}, variants={variants[:5]}")
    trace["normalization"] = {
        s: {"canonical": canonicalize(s), "stripped": strip_version(s)}
        for s in jd_data.get("tech_skills", [])
    }

    # ── Stage 4: Resume chunking ────────────────────────────────────────────
    _section("STAGE 4 — Resume chunking")
    chunks = _split_chunks(resume_text)
    print(f"  {len(chunks)} chunks generated")
    for i, c in enumerate(chunks[:10]):
        print(f"  [{i}] {c[:100]}")
    if len(chunks) > 10:
        print(f"  … {len(chunks) - 10} more chunks")
    trace["resume_chunks"] = chunks

    # ── Stage 5: Per-item match tracing ─────────────────────────────────────
    _section("STAGE 5 — Per-item match trace")
    resume_lower = resume_text.lower()
    match_trace = {}

    all_items = [
        ("tech_skills", s) for s in jd_data.get("tech_skills", [])
    ] + [
        ("soft_skills", s) for s in jd_data.get("soft_skills", [])
    ] + [
        ("amazon_lps", s) for s in jd_data.get("amazon_lps", [])
    ] + [
        ("behavioral_traits", s) for s in jd_data.get("behavioral_traits", [])
    ]

    for cat, item in all_items:
        exact = _is_exact(item, resume_lower)
        implied, trigger = (False, "") if exact else _is_implied(item, resume_lower)
        status = "EXACT" if exact else ("IMPLIED" if implied else "SEMANTIC/MISSING")
        detail = f" ← '{trigger}'" if implied else ""
        print(f"  [{cat}] {item!r:40s} → {status}{detail}")
        match_trace[f"{cat}::{item}"] = {
            "exact": exact,
            "implied": implied,
            "trigger": trigger,
            "status": status,
        }
    trace["match_trace"] = match_trace

    # ── Stage 6: Full comparison ─────────────────────────────────────────────
    _section("STAGE 6 — Comparison output")
    result = compare(jd_data, resume_text)
    print(f"\n  Overall: {result['overall_score']}% — {result['grade']}")
    print(f"  Model:   {result['model_used']}")
    for cat, res in result["categories"].items():
        n_exact   = sum(1 for m in res["matched"] if m["match_type"] == "exact")
        n_implied = sum(1 for p in res["partial"] if p["match_type"] == "implied")
        n_sem     = sum(1 for p in res["partial"] if p["match_type"] in ("close", "semantic"))
        n_miss    = len(res["missing"])
        print(f"  {cat:25s} score={res['score']:3d}%  "
              f"exact={n_exact} implied={n_implied} semantic={n_sem} missing={n_miss}")
        for m in res["missing"]:
            print(f"    MISSING: {m}")
    print()
    for s in result["suggestions"]:
        tag = {"critical": "⛔", "warning": "⚠️", "lp": "★", "info": "💡", "success": "✅"}.get(s["type"], "•")
        print(f"  {tag} {s['text']}")
    trace["comparison"] = result

    # ── Save to file ─────────────────────────────────────────────────────────
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        print(f"\n  Full trace written → {out_path}")

    return trace


# ---------------------------------------------------------------------------
# Built-in demo
# ---------------------------------------------------------------------------

DEMO_JD = """
Software Engineer — Backend

We are looking for an experienced Backend Engineer with strong Python skills
and hands-on experience with Node.js. You should be proficient in PostgreSQL
and have working knowledge of Redis and Kafka. Experience with Kubernetes (k8s)
is required. Familiarity with CI/CD pipelines and GitHub Actions is a plus.
We value engineers with experience in distributed systems, microservices
architecture, and infrastructure as code (Terraform).

You will design and build scalable backend services, lead cross-functional
engineering teams, and mentor junior engineers. We expect you to take ownership
of your work, deliver results consistently, and dive deep into production issues.

Requirements:
- 5+ years of experience
- Bachelor's degree in Computer Science or related field
- Experience with AWS (EC2, RDS, S3, Lambda)
- Strong proficiency in Python 3 and SQL
"""

DEMO_RESUME = """
Senior Software Engineer — Stripe (2020–present)
• Built high-throughput payment APIs in FastAPI (Python) serving 40K RPS
• Managed containerized workloads on EKS using Helm charts
• Set up GitHub Actions CI/CD pipelines and ArgoCD for zero-downtime deployments
• Wrote Terraform modules for provisioning AWS VPCs and ECS clusters
• Designed NodeJS microservices for real-time event processing using SQS
• Spearheaded 35% latency reduction via Aurora Postgres query optimization
• Drove root-cause investigations on P0 incidents, cutting MTTR from 3h to 30min
• Mentored 4 junior engineers, two of whom were promoted to mid-level

Software Engineer — Lyft (2017–2020)
• Built real-time data pipelines with PySpark on EMR
• Delivered core matching algorithm refactor 2 weeks ahead of schedule
• Used Redis for session caching and Elasticsearch for search indexing

Skills: Python, JavaScript, TypeScript, Go, PostgreSQL, MySQL, MongoDB,
        Redis, Kafka, Docker, Kubernetes, Terraform, AWS, GitHub Actions
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SYSkeywords pipeline debugger")
    parser.add_argument("--jd",     help="Path to JD text file")
    parser.add_argument("--resume", help="Path to resume text file")
    parser.add_argument("--out",    help="Write full JSON trace to this file")
    parser.add_argument("--demo",   action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    if args.demo:
        debug(DEMO_JD, DEMO_RESUME, args.out)
    elif args.jd and args.resume:
        debug(_read(args.jd), _read(args.resume), args.out)
    else:
        parser.print_help()
        sys.exit(1)
