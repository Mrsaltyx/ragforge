"""Embeddings — interface unique, deux backends interchangeables.

- HashingEmbedder : déterministe, sans téléchargement. Pour les tests et la
  démo hors-ligne. (Une baseline de recherche doit tourner sans GPU ni clé.)
- E5Embedder : sentence-transformers (multilingual-e5-small), adapté au français,
  avec les préfixes "query: "/"passage: " exigés par la famille E5.
"""

import hashlib
import re
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9àâäéèêëîïôöùûüç]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray: ...


class HashingEmbedder:
    """Vecteurs par hachage de tokens (signé), normalisés L2. Déterministe."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class E5Embedder:
    """Embeddings sémantiques multilingues via sentence-transformers (lazy load)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", dim: int = 384):
        from sentence_transformers import SentenceTransformer  # import différé : dépendance lourde

        self.dim = dim
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        prefix = "query: " if kind == "query" else "passage: "
        emb = self._model.encode([prefix + t for t in texts], normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


def build_embedder(backend: str, model: str, dim: int) -> Embedder:
    if backend == "e5":
        return E5Embedder(model_name=model, dim=dim)
    return HashingEmbedder(dim=dim)
