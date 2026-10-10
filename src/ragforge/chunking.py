"""Découpage structure-aware : paragraphes → phrases → recouvrement contrôlé.

Pourquoi pas des chunks de taille fixe : couper au milieu d'une phrase casse
la sémantique et dégrade le retrieval. On accumule des phrases entières jusqu'à
la taille cible, puis le chunk suivant reprend les phrases couvrant les
`overlap` derniers caractères du chunk précédent.
"""

import re
from dataclasses import dataclass, field

SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _sentences(text: str) -> list[str]:
    """Découpe en phrases en traitant les retours à la ligne comme frontières."""
    parts = SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _headings(text: str) -> list[str]:
    """Extrait les titres markdown (# …) — métadonnée utile pour le debug."""
    return [line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")]


def chunk_text(text: str, source: str, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Découpe un document en chunks de phrases entières avec recouvrement.

    Args:
        text: contenu brut du document.
        source: nom du document (préfixe des IDs de chunks).
        size: taille cible en caractères.
        overlap: recouvrement en caractères entre chunks successifs.

    Returns:
        Liste de chunks avec métadonnées (source, index, titres).
    """
    sentences = _sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(sentences):
        # accumulate phrases entières jusqu'à la taille cible
        current: list[str] = []
        length = 0
        j = i
        while j < len(sentences) and length + len(sentences[j]) + 1 <= size:
            current.append(sentences[j])
            length += len(sentences[j]) + 1
            j += 1

        if not current:
            # phrase unique plus longue que la taille cible → coupe dure (fallback)
            body = sentences[i][:size]
            chunks.append(_make_chunk(source, len(chunks), body))
            i += 1
            continue

        chunks.append(_make_chunk(source, len(chunks), " ".join(current)))
        if j >= len(sentences):
            break

        # recouvrement : reculer pour reprendre les phrases couvrant `overlap` caractères
        acc = 0
        k = j - 1
        while k > i and acc + len(sentences[k]) + 1 <= overlap:
            acc += len(sentences[k]) + 1
            k -= 1
        i = k

    return chunks


def _make_chunk(source: str, index: int, body: str) -> Chunk:
    return Chunk(
        id=f"{source}::chunk-{index}",
        text=body,
        metadata={
            "source": source,
            "chunk_index": index,
            "headings": _headings(body),
            "char_len": len(body),
        },
    )
