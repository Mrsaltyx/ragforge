"""LLM — client compatible OpenAI (Ollama, OpenAI, GLM/Z.AI, vLLM…) + repli extractif.

Le pipeline ne doit jamais dépendre d'un provider unique : tout endpoint
compatible /chat/completions fonctionne. Sans clé ni endpoint configuré,
le repli extractif renvoie les chunks verbatim avec une mention explicite :
la démo reste utilisable à zéro config.
"""

import re
from typing import Protocol

import httpx

from .vectorstore import ScoredChunk

SYSTEM_PROMPT = """Tu es un assistant documentaire précis.
Réponds UNIQUEMENT à partir du contexte fourni, entre balises <contexte> et </contexte>.
Cite tes sources avec les balises [S1], [S2], … correspondant aux passages du contexte.
Si le contexte ne permet pas de répondre, dis-le clairement en une phrase.
Réponds en français, de manière concise et structurée."""

CITATION_RE = re.compile(r"\[S(\d+)\]")


class BaseLLM(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, timeout: float):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class ExtractiveLLM:
    """Repli sans LLM : concatène les chunks verbatim, citations [S1], [S2] explicites."""

    def __init__(self, sources: list[ScoredChunk]):
        self._sources = sources

    def complete(self, messages: list[dict]) -> str:
        lines = [
            "*(Mode extractif — aucun LLM configuré. Extraits verbatim les plus pertinents :)*",
            "",
        ]
        for i, scored in enumerate(self._sources, start=1):
            lines.append(f"[S{i}] ({scored.chunk.metadata.get('source', '?')}) {scored.chunk.text}")
        return "\n\n".join(lines)


def extract_citations(answer: str, retrieved: list[ScoredChunk]) -> list[str]:
    """Balise → chunk réellement récupéré. Détecte les citations hallucinées."""
    cited: list[str] = []
    for match in CITATION_RE.finditer(answer):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(retrieved):
            chunk_id = retrieved[idx].chunk.id
            if chunk_id not in cited:
                cited.append(chunk_id)
    return cited


def build_llm(
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    fallback_sources: list[ScoredChunk],
) -> BaseLLM:
    if backend == "extractive":
        return ExtractiveLLM(fallback_sources)
    try:
        # sonde : l'endpoint répond-il ? sinon repli extractif silencieux
        probe = OpenAICompatLLM(base_url, api_key, model, temperature, timeout=5.0)
        probe.complete([{"role": "user", "content": "ping"}])
        return OpenAICompatLLM(base_url, api_key, model, temperature, timeout)
    except httpx.HTTPError:  # endpoint injoignable → repli extractif explicite
        return ExtractiveLLM(fallback_sources)
