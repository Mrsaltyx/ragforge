"""Configuration centralisée — toutes les valeurs surchargées par variables d'environnement (préfixe RAGFORGE_)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGFORGE_", env_file=".env", extra="ignore")

    # Stockage
    data_dir: Path = Path("./data")

    # Chunking
    chunk_size: int = 800        # caractères cible par chunk
    chunk_overlap: int = 120     # recouvrement entre chunks successifs

    # Embeddings (backend numpy = hashage déterministe, sans téléchargement)
    embedding_backend: str = "hashing"   # hashing | e5
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # Retrieval hybride
    vector_backend: str = "numpy"        # numpy | chroma
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60

    # Reranking (activable — ne s'active que si l'eval le justifie)
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    final_top_k: int = 5

    # LLM (client compatible OpenAI : Ollama, OpenAI, GLM/Z.AI, vLLM…)
    llm_backend: str = "openai_compat"   # openai_compat | extractive
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.2
    llm_timeout: float = 60.0
