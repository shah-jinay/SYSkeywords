"""Pytest configuration — adds src/ to sys.path so tests import tailor directly."""
import sys
import os
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Redirect DB and out dirs to a temp directory for every test."""
    import tailor.config as cfg_module

    db_dir = tmp_path / "data" / "db"
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)

    new_settings = cfg_module.Settings(
        data_dir=tmp_path / "data",
        db_dir=db_dir,
        library_db=db_dir / "library.sqlite",
        history_db=db_dir / "history.sqlite",
        out_dir=tmp_path / "out",
        master_resume=ROOT / "data" / "master_resume.txt",
        role_resumes_dir=ROOT / "data" / "role_resumes",
    )
    monkeypatch.setattr(cfg_module, "settings", new_settings)

    for mod_name in ("tailor.library", "tailor.embeddings", "tailor.tailor",
                     "tailor.generator", "tailor.learner"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "settings"):
            monkeypatch.setattr(mod, "settings", new_settings)
