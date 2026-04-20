# nlp-gateway/routers/tts.py
#
# TTS 文字转语音 API

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import os
import logging

from services.tts_service import synthesize

router = APIRouter()

logger = logging.getLogger("nlp_gateway.tts_router")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False


class TtsReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    voice: Optional[str] = "nova"
    traceId: Optional[str] = None


@router.post("/speak")
async def speak(req: TtsReq):
    """
    文字转语音，返回 MP3 音频流。
    """
    trace_id = req.traceId or "-"
    logger.info("[traceId=%s] /tts/speak voice=%s text=%s", trace_id, req.voice, req.text[:60])

    try:
        audio_bytes = synthesize(req.text, req.voice or "nova")
    except RuntimeError as e:
        logger.error("[traceId=%s] /tts/speak error: %s", trace_id, e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("[traceId=%s] /tts/speak error: %s", trace_id, e)
        raise HTTPException(status_code=500, detail="TTS generation failed")

    logger.info("[traceId=%s] /tts/speak => %.1fKB", trace_id, len(audio_bytes) / 1024)

    return Response(content=audio_bytes, media_type="audio/mpeg")
