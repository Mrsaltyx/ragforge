"""Vector stores — interface unique, deux backends interchangeables.

- NumpyStore : cosinus en matriciel, persistance .npz — zéro dépendance lourde,
  parfait pour démarrer et pour les tests.
- ChromaStore : ChromaDB persistant, le bon choix quand le corpus grandit.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    origin: str = "dense"  # dense | bm25 | rrf | rerank — traçabilité du pipeline


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]: ...
    def delete_by_source(self, source: str) -> None: ...
    def list_sources(self) -> list[str]: ...


class NumpyStore:
    """Store dense en mémoire + persistance simple (data/vectors.npz + meta.json)."""

    def __init__(self, persist_dir: Path, dim: int):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._load()

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for chunk in chunks:
            self._ids.append(chunk.id)
            self._chunks[chunk.id] = chunk
        self._matrix = np.vstack([self._matrix, vectors]) if len(self._ids) > len(vectors) else vectors
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if self._matrix.shape[0] == 0:
            return []
        scores = self._matrix @ query_vector  # vecteurs déjà normalisés L2
        order = np.argsort(-scores)[:top_k]
        return [ScoredChunk(self._chunks[self._ids[i]], float(scores[i]), "dense") for i in order]

    def delete_by_source(self, source: str) -> None:
        keep = [i for i, cid in enumerate(self._ids) if self._chunks[cid].metadata.get("source") != source]
        self._ids = [self._ids[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else self._matrix[:0]
        self._chunks = {cid: c for cid, c in self._chunks.items() if c.metadata.get("source") != source}
        self._save()

    def list_sources(self) -> list[str]:
        return sorted({c.metadata.get("source", "") for c in self._chunks.values()})

    def _save(self) -> None:
        np.savez(self.persist_dir / "vectors.npz", matrix=self._matrix)
        meta = {cid: {"text": c.text, "metadata": c.metadata} for cid, c in self._chunks.items()}
        (self.persist_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        vec_file = self.persist_dir / "vectors.npz"
        meta_file = self.persist_dir / "meta.json"
        if vec_file.exists() and meta_file.exists():
            self._matrix = np.load(vec_file)["matrix"]
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self._chunks = {
                cid: Chunk(id=cid, text=v["text"], metadata=v["metadata"]) for cid, v in meta.items()
            }
            self._ids = list(self._chunks.keys())


class ChromaStore:
    """Store dense ChromaDB persistant (nécessite l'extra : pip install .[chroma])."""

    def __init__(self, persist_dir: Path, dim: int):
        import chromadb  # import différé : dépendance lourde

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection("ragforge", metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._col.add(
            ids=[c.id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.metadata.get("source", ""), "chunk_index": c.metadata.get("chunk_index", 0)} for c in chunks],
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        res = self._col.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
        out = []
        for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(id=cid, text=text, metadata=dict(meta))
            out.append(ScoredChunk(chunk, 1.0 - float(dist), "dense"))
        return out

    def delete_by_source(self, source: str) -> None:
        self._col.delete(where={"source": source})

    def list_sources(self) -> list[str]:
        return sorted({m["source"] for m in self._col.get(include=["metadatas"])["metadatas"]})


def build_vector_store(backend: str, data_dir: Path, dim: int) -> VectorStore:
    if backend == "chroma":
        return ChromaStore(data_dir / "chroma", dim)
    return NumpyStore(data_dir / "numpy", dim)
