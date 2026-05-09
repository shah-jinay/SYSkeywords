"""Record accept/reject/edit decisions and compute Bayesian ranking bonus."""
from __future__ import annotations
from typing import Optional

_BETA_A, _BETA_B = 2, 2  # Beta(2,2) prior


def record_decision(
    bullet_id: int,
    jd_slug: str,
    decision: str,
    edited_text: str = "",
    session_id: Optional[int] = None,
) -> int:
    """Record a decision. Returns the session_id used."""
    from .library import _conn, _hist
    with _conn(_hist()) as db:
        if session_id is None:
            cur = db.execute("INSERT INTO sessions (jd_slug) VALUES (?)", (jd_slug,))
            session_id = cur.lastrowid
        db.execute(
            "INSERT INTO decisions (session_id, bullet_id, decision, edited_text)"
            " VALUES (?,?,?,?)",
            (session_id, bullet_id, decision, edited_text),
        )
    return session_id


def bayesian_bonus(bullet_id: int) -> float:
    """Return score bonus in [0, 0.15] using Beta posterior over accept/reject counts."""
    from .library import _conn, _hist
    with _conn(_hist()) as db:
        row = db.execute(
            """
            SELECT
                SUM(CASE WHEN decision='accept' THEN 1 ELSE 0 END) as accepts,
                SUM(CASE WHEN decision='reject' THEN 1 ELSE 0 END) as rejects
            FROM decisions WHERE bullet_id = ?
            """,
            (bullet_id,),
        ).fetchone()

    if not row or not (row["accepts"] or row["rejects"]):
        return 0.0

    alpha = _BETA_A + (row["accepts"] or 0)
    beta  = _BETA_B + (row["rejects"] or 0)
    mean    = alpha / (alpha + beta)
    neutral = _BETA_A / (_BETA_A + _BETA_B)
    return min(max(mean - neutral, 0.0), 0.15)


def get_stats(jd_slug: Optional[str] = None) -> list[dict]:
    """Return per-bullet accept/reject counts, sorted by accepts desc."""
    from .library import _conn, _hist, get_all_bullets
    bullets = {b["id"]: b for b in get_all_bullets()}

    with _conn(_hist()) as db:
        q = """
            SELECT d.bullet_id,
                   SUM(CASE WHEN d.decision='accept' THEN 1 ELSE 0 END) as accepts,
                   SUM(CASE WHEN d.decision='reject' THEN 1 ELSE 0 END) as rejects
            FROM decisions d
            JOIN sessions s ON d.session_id = s.id
        """
        params: list = []
        if jd_slug:
            q += " WHERE s.jd_slug = ?"
            params.append(jd_slug)
        q += " GROUP BY d.bullet_id ORDER BY accepts DESC"
        rows = db.execute(q, params).fetchall()

    result = []
    for r in rows:
        b = bullets.get(r["bullet_id"], {})
        result.append({
            "bullet_id": r["bullet_id"],
            "accepts": r["accepts"],
            "rejects": r["rejects"],
            "company": b.get("company", ""),
            "text": b.get("text", ""),
        })
    return result
