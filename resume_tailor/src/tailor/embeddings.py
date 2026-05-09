"""Sentence-transformer embeddings with SQLite cache."""
from __future__ import annotations
import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from .config import settings
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.embedding_model)
        except Exception:
            _model = False
    return _model if _model is not False else None


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns (N, D) float32 array."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    if model:
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype(np.float32)
    return _tfidf_embed(texts)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def get_or_compute_bullet_embeddings() -> dict[int, np.ndarray]:
    """Return {bullet_id: embedding} for all bullets, computing any missing."""
    from .library import get_all_bullets, get_embedding, save_embedding, get_bullets_without_embeddings

    missing = get_bullets_without_embeddings()
    if missing:
        vecs = embed([b["text"] for b in missing])
        for b, vec in zip(missing, vecs):
            save_embedding(b["id"], vec)

    result: dict[int, np.ndarray] = {}
    for b in get_all_bullets():
        vec = get_embedding(b["id"])
        if vec is not None:
            result[b["id"]] = vec
    return result


def _tfidf_embed(texts: list[str]) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        mat = TfidfVectorizer(max_features=384).fit_transform(texts).toarray()
        return mat.astype(np.float32)
    except Exception:
        return np.zeros((len(texts), 384), dtype=np.float32)
