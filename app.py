import os
import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file

from extractor import extract
from comparator import compare

# ── tailor package path + settings (must happen before any tailor import) ──
_TAILOR_ROOT = Path(__file__).parent / "resume_tailor"
_TAILOR_SRC  = _TAILOR_ROOT / "src"
if str(_TAILOR_SRC) not in sys.path:
    sys.path.insert(0, str(_TAILOR_SRC))

os.environ.setdefault("TAILOR_DATA_DIR",           str(_TAILOR_ROOT / "data"))
os.environ.setdefault("TAILOR_DB_DIR",             str(_TAILOR_ROOT / "data" / "db"))
os.environ.setdefault("TAILOR_LIBRARY_DB",         str(_TAILOR_ROOT / "data" / "db" / "library.sqlite"))
os.environ.setdefault("TAILOR_HISTORY_DB",         str(_TAILOR_ROOT / "data" / "db" / "history.sqlite"))
os.environ.setdefault("TAILOR_PROMPTS_DIR",        str(_TAILOR_ROOT / "prompts"))
os.environ.setdefault("TAILOR_OUT_DIR",            str(_TAILOR_ROOT / "out"))
os.environ.setdefault("TAILOR_MASTER_RESUME",      str(_TAILOR_ROOT / "data" / "master_resume.txt"))
os.environ.setdefault("TAILOR_ROLE_RESUMES_DIR",   str(_TAILOR_ROOT / "data" / "role_resumes"))

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jd_to_tiers(jd_data: dict) -> dict:
    """Map extractor output → CRITICAL / IMPORTANT / NICE_TO_HAVE tiers."""
    def dedup(lst: list) -> list:
        seen: set = set()
        out: list = []
        for x in lst:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    tech     = jd_data.get("tech_skills", [])
    inferred = jd_data.get("inferred_domains", [])
    resp     = jd_data.get("responsibilities", [])[:6]
    soft     = jd_data.get("soft_skills", [])
    lps      = jd_data.get("amazon_lps", [])
    bt       = jd_data.get("behavioral_traits", [])

    return {
        "CRITICAL":     dedup(tech)[:15],
        "IMPORTANT":    dedup(inferred + resp + soft)[:12],
        "NICE_TO_HAVE": dedup(lps + bt)[:8],
    }


# ---------------------------------------------------------------------------
# Existing routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract_keywords():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(extract(text))


@app.route("/compare", methods=["POST"])
def compare_resume():
    data        = request.get_json(force=True)
    jd_data     = data.get("jd_data")
    resume_text = (data.get("resume_text") or "").strip()
    if not jd_data:
        return jsonify({"error": "Extract the JD first, then compare."}), 400
    if not resume_text:
        return jsonify({"error": "No resume text provided."}), 400
    return jsonify(compare(jd_data, resume_text))


# ---------------------------------------------------------------------------
# Tailor routes
# ---------------------------------------------------------------------------

@app.route("/api/sources", methods=["GET"])
def get_sources():
    """Return available resume source files from the bullet library DB."""
    try:
        from tailor.library import init_dbs, list_sources
        init_dbs()
        sources = list_sources()
        return jsonify({"sources": [s["source_file"] for s in sources if s["source_file"]]})
    except Exception as e:
        return jsonify({"sources": [], "error": str(e)})


@app.route("/api/suggest-resume", methods=["POST"])
def suggest_resume():
    """Score each ingested source against JD keywords and return ranked suggestions."""
    data    = request.get_json(force=True)
    jd_data = data.get("jd_data") or {}
    tiered  = _jd_to_tiers(jd_data)

    try:
        from tailor.library import init_dbs, count_bullets
        from tailor.matcher import match_bullets_to_jd
        init_dbs()
        if count_bullets() == 0:
            return jsonify({"suggestions": []})

        matches = match_bullets_to_jd(tiered, top_k=200)

        # Aggregate score per source: sum of top-10 bullet scores
        from collections import defaultdict
        source_scores: dict[str, list[float]] = defaultdict(list)
        for m in matches:
            source_scores[m.source_file].append(m.score)

        ranked = []
        for src, scores in source_scores.items():
            top10 = sorted(scores, reverse=True)[:10]
            ranked.append({"source": src, "score": round(sum(top10), 3)})

        ranked.sort(key=lambda x: -x["score"])
        # Normalise to 0-100 relative to best score
        best = ranked[0]["score"] if ranked else 1
        for r in ranked:
            r["pct"] = round(r["score"] / best * 100)

        return jsonify({"suggestions": ranked})
    except Exception as e:
        return jsonify({"suggestions": [], "error": str(e)})


@app.route("/api/resume-sources", methods=["GET"])
def get_resume_sources():
    """Return ingested resume sources with bullet counts."""
    try:
        from tailor.library import init_dbs, list_sources
        init_dbs()
        rows = list_sources()
        result = [{"source": r["source_file"], "count": r["count"]}
                  for r in rows if r["source_file"]]
        return jsonify({"sources": result})
    except Exception as e:
        return jsonify({"sources": [], "error": str(e)})


@app.route("/api/resume-text/<source>", methods=["GET"])
def get_resume_text(source):
    """Return raw text of an ingested resume file."""
    from tailor.config import settings
    master = Path(str(settings.master_resume))
    # "master" source maps to master_resume.txt
    if source == "master" and master.exists():
        text = master.read_text(encoding="utf-8-sig")
        return jsonify({"source": source, "text": text, "chars": len(text)})
    role_path = Path(str(settings.role_resumes_dir)) / f"{source}.txt"
    if role_path.exists():
        text = role_path.read_text(encoding="utf-8-sig")
        return jsonify({"source": source, "text": text, "chars": len(text)})
    return jsonify({"error": f"Resume file '{source}' not found"}), 404


@app.route("/tailor", methods=["POST"])
def generate_resume():
    data    = request.get_json(force=True)
    jd_text = (data.get("jd_text") or "").strip()
    role    = (data.get("role") or "").strip()

    if not jd_text:
        return jsonify({"error": "No JD text provided"}), 400

    # Lazy imports — keeps startup fast
    from tailor.library import init_dbs, count_bullets
    from tailor.matcher import match_bullets_to_jd
    from tailor.tailor import assemble_doc
    from tailor.generator import generate_pdf

    init_dbs()

    if count_bullets() == 0:
        return jsonify({
            "error": (
                "Resume library is empty.\n"
                "Run once:  cd resume_tailor && PYTHONPATH=src python -m tailor.cli ingest"
            )
        }), 400

    # Step 1 — extract + tier JD keywords
    jd_data        = extract(jd_text)
    jd_data["_slug"] = "web"
    tiered         = _jd_to_tiers(jd_data)

    # Step 2 — match bullets (more candidates so each company gets good coverage)
    matches = match_bullets_to_jd(tiered, top_k=80, source_hint=role)

    # Step 3 — assemble full-structure doc (header, skills, exp grouped by company, edu)
    doc = assemble_doc(matches, tiered, jd_data, role=role)

    if not doc.experience and not doc.bullets:
        return jsonify({"error": "No matching bullets found. Try a different role."}), 400

    # Step 4 — generate PDF
    pdf_path = generate_pdf(doc)
    filename = f"resume_{role or 'tailored'}.pdf"

    return send_file(
        str(pdf_path),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
