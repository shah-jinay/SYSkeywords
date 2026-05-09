"""Configuration via pydantic-settings — reads TAILOR_* env vars or .env."""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path("data")
    db_dir: Path = Path("data/db")
    library_db: Path = Path("data/db/library.sqlite")
    history_db: Path = Path("data/db/history.sqlite")
    prompts_dir: Path = Path("prompts")
    out_dir: Path = Path("out")
    master_resume: Path = Path("data/master_resume.txt")
    role_resumes_dir: Path = Path("data/role_resumes")
    embedding_model: str = "all-MiniLM-L6-v2"
    semantic_threshold: float = 0.55

    model_config = {"env_file": ".env", "env_prefix": "TAILOR_"}


settings = Settings()
