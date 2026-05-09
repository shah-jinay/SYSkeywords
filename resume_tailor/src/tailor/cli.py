"""tailor CLI — entry point via `python -m tailor.cli` or `tailor` command."""
from __future__ import annotations
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="tailor", help="Local resume tailoring against JD keywords.", no_args_is_help=True)
console = Console()


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Ingest a specific file"),
    role: Optional[str] = typer.Option(None, "--role", "-r", help="Ingest data/role_resumes/<ROLE>.txt"),
) -> None:
    """
    Load resume bullets into the library.

    With no flags: ingests master_resume.txt + every file in data/role_resumes/.
    --role python: ingests only data/role_resumes/python.txt.
    --file <path>: ingests a specific file.
    """
    from .library import init_dbs, list_sources
    from .config import settings
    from .ingestion import ingest_resume, ingest_all

    init_dbs()

    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        n = ingest_resume(file, source_file=file.stem)
        console.print(f"[green]Ingested {n} bullets from {file}[/green]")

    elif role:
        role_path = Path(settings.role_resumes_dir) / f"{role}.txt"
        if not role_path.exists():
            candidates = list(Path(settings.role_resumes_dir).glob("*.txt"))
            matches = [p for p in candidates if p.stem.lower() == role.lower()]
            if matches:
                role_path = matches[0]
            else:
                console.print(f"[red]Role resume not found: {role_path}[/red]")
                console.print(f"Available: {[p.stem for p in candidates]}")
                raise typer.Exit(1)
        n = ingest_resume(role_path, source_file=role_path.stem)
        console.print(f"[green]Ingested {n} bullets from {role_path}[/green]")

    else:
        counts = ingest_all(
            master_resume=Path(settings.master_resume),
            role_resumes_dir=Path(settings.role_resumes_dir),
        )
        total = sum(counts.values())
        console.print(f"\n[bold green]Ingested {total} bullets total[/bold green]")
        t = Table(show_header=True, header_style="bold")
        t.add_column("Source", style="cyan")
        t.add_column("Bullets", justify="right")
        for src, cnt in counts.items():
            t.add_row(src, str(cnt))
        console.print(t)

    sources = list_sources()
    if sources:
        console.print("\n[dim]Library now contains:[/dim]")
        for s in sources:
            console.print(f"  [dim]{s['source_file'] or '(untagged)':15s}  {s['count']} bullets[/dim]")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    jd: str = typer.Option(..., help="JD text or path to a .json file under data/jd/"),
    top: int = typer.Option(20, help="Number of top matches to show"),
    role: Optional[str] = typer.Option(None, "--role", help="Source hint for boost"),
) -> None:
    """Show top bullet matches for a JD without generating output."""
    import json
    from .library import init_dbs
    from .matcher import match_bullets_to_jd
    from .config import settings

    init_dbs()

    jd_path = Path(settings.data_dir) / "jd" / f"{jd}.json"
    if jd_path.exists():
        with open(jd_path) as f:
            jd_raw = json.load(f)
        jd_text = jd_raw.get("jd_text", "")
    else:
        jd_text = jd

    try:
        sys_path_root = Path(__file__).parent.parent.parent.parent
        import sys
        if str(sys_path_root) not in sys.path:
            sys.path.insert(0, str(sys_path_root))
        from extractor import extract  # type: ignore[import]
        jd_data = extract(jd_text)
    except ImportError:
        jd_data = {"tech_skills": [], "inferred_domains": [], "responsibilities": []}

    tiered = {
        "CRITICAL": jd_data.get("tech_skills", [])[:15],
        "IMPORTANT": (
            jd_data.get("inferred_domains", [])
            + jd_data.get("responsibilities", [])[:6]
        )[:12],
        "NICE_TO_HAVE": []
    }

    matches = match_bullets_to_jd(tiered, top_k=top, source_hint=role or "")

    t = Table(title=f"Top {top} matches", show_lines=False)
    t.add_column("Score", width=6, style="cyan")
    t.add_column("Tier", width=12, style="yellow")
    t.add_column("Source", width=10, style="dim")
    t.add_column("Company", width=22)
    t.add_column("Keyword", width=20, style="magenta")
    t.add_column("Bullet", no_wrap=False)

    for m in matches:
        t.add_row(
            f"{m.score:.3f}",
            m.tier,
            m.source_file or "",
            m.company or "—",
            m.keyword,
            m.text[:90] + ("…" if len(m.text) > 90 else ""),
        )
    console.print(t)


# ---------------------------------------------------------------------------
# draft
# ---------------------------------------------------------------------------

@app.command()
def draft(
    jd: str = typer.Option(..., help="JD slug (data/jd/<slug>.json)"),
    role: Optional[str] = typer.Option(None, "--role", help="Target role label"),
    bullets_count: int = typer.Option(8, "--bullets", help="Number of bullets"),
    pdf: bool = typer.Option(True, help="Generate PDF (--no-pdf to skip)"),
    out: Optional[Path] = typer.Option(None, help="Custom PDF output path"),
) -> None:
    """Select top bullets for a JD and optionally generate a PDF."""
    from .library import init_dbs
    from .tailor import draft_resume
    from .generator import generate_pdf

    init_dbs()
    result = draft_resume(jd_slug=jd, role=role, target_bullets=bullets_count)

    console.print(f"\n[bold]Draft for '{jd}'[/bold]  ({len(result['bullets'])} bullets)\n")
    for i, b in enumerate(result["bullets"], 1):
        issues = b.get("issues", [])
        flag = "[red][!][/red] " if issues else "[green][ok][/green] "
        console.print(f"  {i}. {flag}[dim]{b['score']:.3f}[/dim] [cyan]{b['company'] or '—'}[/cyan]")
        console.print(f"     {b['text']}")
        for issue in issues:
            console.print(f"     [red]  rule: {issue}[/red]")
        console.print()

    if pdf:
        pdf_path = generate_pdf(result, out_path=out)
        console.print(f"[green bold]PDF -> {pdf_path}[/green bold]")


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

@app.command()
def review(
    jd: str = typer.Option(..., help="JD slug"),
    bullets_count: int = typer.Option(15, "--bullets", help="Number of bullets to review"),
) -> None:
    """Interactive accept/reject/edit session for learning."""
    import questionary
    from .library import init_dbs
    from .tailor import draft_resume
    from .learner import record_decision

    init_dbs()
    result = draft_resume(jd_slug=jd, target_bullets=bullets_count)
    console.print(f"\n[bold]Review session for '{jd}'[/bold] — {len(result['bullets'])} bullets\n")

    session_id = None
    for b in result["bullets"]:
        console.rule()
        console.print(f"[bold]{b['text']}[/bold]")
        console.print(f"[dim]company={b['company']}  score={b['score']:.3f}  keyword={b['keyword']}[/dim]\n")

        decision = questionary.select("Decision:", choices=["accept", "reject", "edit", "skip"]).ask()
        if decision is None or decision == "skip":
            continue

        edited = ""
        if decision == "edit":
            edited = questionary.text("Edit bullet:", default=b["text"]).ask() or ""

        session_id = record_decision(b["id"], jd, decision, edited, session_id=session_id)
        console.print(f"  [green]Recorded: {decision}[/green]\n")

    console.print("[bold green]Review complete.[/bold green]")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@app.command()
def stats(
    jd: Optional[str] = typer.Option(None, "--jd", help="Filter by JD slug"),
) -> None:
    """Show per-bullet decision statistics."""
    from .learner import get_stats

    rows = get_stats(jd_slug=jd)
    if not rows:
        console.print("[dim]No decisions recorded yet. Run `tailor review` first.[/dim]")
        return

    t = Table(title="Bullet Stats" + (f" — {jd}" if jd else ""))
    t.add_column("Accepts", width=8, style="green")
    t.add_column("Rejects", width=8, style="red")
    t.add_column("Company", width=22)
    t.add_column("Bullet", no_wrap=False)

    for r in rows:
        t.add_row(
            str(r["accepts"]),
            str(r["rejects"]),
            r["company"] or "—",
            r["text"][:80] + ("…" if len(r["text"]) > 80 else ""),
        )
    console.print(t)


if __name__ == "__main__":
    app()
