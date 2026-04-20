"""
Local embedding service using BAAI/bge-small-zh-v1.5
- 512-dim vectors, ~95MB model
- Loaded once at startup, reused for all requests
"""
import logging
import os
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("nlp_gateway.embedding")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded. dim=%d", _model.get_sentence_embedding_dimension())
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
