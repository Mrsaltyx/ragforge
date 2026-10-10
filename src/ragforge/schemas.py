"""Contrats d'API — schémas Pydantic en entrée ET en sortie."""

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Un chunk cité dans la réponse."""

    rank: int
    chunk_id: str
    document: str
    score: float
    excerpt: str = Field(description="Début du chunk cité (~300 caractères)")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    cited_sources: list[str] = Field(description="IDs de chunks réellement cités dans la réponse")
    citation_faithful: bool = Field(description="Toutes les citations renvoient à un chunk récupéré")
    timings_ms: dict[str, float]


class IngestResponse(BaseModel):
    document: str
    chunks_created: int


class EvalResult(BaseModel):
    metric: str
    value: float
    detail: str = ""
