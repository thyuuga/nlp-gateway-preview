# nlp-gateway/routers/emotion.py
#
# Emotion 分析 API

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import logging

from services.emotion_service import predict_emotion

router = APIRouter()

logger = logging.getLogger("nlp_gateway.emotion_router")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False


class EmotionCandidate(BaseModel):
    emotion: str       # 英文标签: neutral/happy/sad/angry/proud/shy
    label_zh: str      # 中文标签: 平常/开心/伤心/生气/得意/害羞
    prob: float


class EmotionReq(BaseModel):
    text: str = Field(..., min_length=1)
    traceId: Optional[str] = None


class EmotionResp(BaseModel):
    emotion: str       # 英文标签
    label_zh: str      # 中文标签
    confidence: float
    candidates: List[EmotionCandidate]


@router.post("/analyze", response_model=EmotionResp)
async def analyze_emotion(req: EmotionReq):
    """
    单句情绪分析

    返回 6 类中的最高概率标签 + 全部候选概率。
    模型不可用时返回 500。
    """
    trace_id = req.traceId or "-"
    logger.info("[traceId=%s] /emotion/analyze text=%s", trace_id, req.text[:60] if req.text else "")

    result = predict_emotion(req.text)

    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Emotion model unavailable")

    logger.info("[traceId=%s] /emotion/analyze => %s (%.3f)", trace_id, result["emotion"], result["confidence"])

    return result
