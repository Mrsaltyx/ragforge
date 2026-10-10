"""Index lexical BM25 — capture les termes exacts que le dense rate (codes, noms, sigles)."""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .vectorstore import ScoredChunk

TOKEN_RE = re.compile(r"[a-z0-9àâäéèêëîïôöùûüç]+")


class BM25Index:
    def __init__(self, persist_file: Path):
        self.persist_file = Path(persist_file)
        self.persist_file.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None
        self._load()

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._rebuild()
        self._save()

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not self._index:
            return []
        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [ScoredChunk(self._chunks[i], float(scores[i]), "bm25") for i in order]

    def delete_by_source(self, source: str) -> None:
        self._chunks = [c for c in self._chunks if c.metadata.get("source") != source]
        self._rebuild()
        self._save()

    def _rebuild(self) -> None:
        corpus = [self._tokenize(c.text) for c in self._chunks]
        self._index = BM25Okapi(corpus) if corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def _save(self) -> None:
        with open(self.persist_file, "wb") as f:
            pickle.dump(self._chunks, f)

    def _load(self) -> None:
        if self.persist_file.exists():
            with open(self.persist_file, "rb") as f:
                self._chunks = pickle.load(f)
            self._rebuild()
