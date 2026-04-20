from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List
import logging
import os

from services.embedding_service import encode

router = APIRouter()

logger = logging.getLogger("nlp_gateway.embed_encode")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False


class EncodeReq(BaseModel):
    texts: List[str]


class EncodeResp(BaseModel):
    embeddings: List[List[float]]


@router.post("/encode", response_model=EncodeResp)
def encode_texts(req: EncodeReq, request: Request):
    trace_id = request.headers.get("x-trace-id") or "-"
    if not req.texts:
        return EncodeResp(embeddings=[])

    logger.info("[traceId=%s] /embed/encode count=%d first=%s",
                trace_id, len(req.texts), req.texts[0][:50] if req.texts[0] else "")

    vectors = encode(req.texts)

    logger.info("[traceId=%s] encoded %d texts, dim=%d",
                trace_id, len(vectors), len(vectors[0]) if vectors else 0)

    return EncodeResp(embeddings=vectors)
