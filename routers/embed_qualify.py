from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional
import os
import logging

from services.message_filter import MessageFilter

router = APIRouter()

# ---- logging ----
logger = logging.getLogger("nlp_gateway.embed_qualify")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False

# ---- 缓存 filter 实例 ----
_filters = {}


def get_filter(lang: str = "zh") -> MessageFilter:
    if lang not in _filters:
        _filters[lang] = MessageFilter(lang)
    return _filters[lang]


# ---- Request/Response Models ----

class FilterMessageReq(BaseModel):
    text: str
    role: str = "user"
    lang: str = "zh"


class FilterMessageResp(BaseModel):
    should_embed: bool
    score: int
    coherence: float
    sentence_type: Optional[str] = None
    elements: List[str] = []
    features: List[str] = []
    penalties: List[str] = []
    reason: str = ""


class BatchFilterReq(BaseModel):
    messages: List[FilterMessageReq]


class BatchFilterResp(BaseModel):
    results: List[FilterMessageResp]


# ---- Endpoints ----

@router.post("/qualify", response_model=FilterMessageResp)
def qualify_message_for_embed(req: FilterMessageReq, request: Request):
    """
    判断消息是否值得做 embedding

    POST /embed/qualify
    """
    trace_id = request.headers.get("x-trace-id") or "-"
    logger.debug("[traceId=%s] /embed/qualify text=%s", trace_id, req.text[:50] if req.text else "")

    try:
        f = get_filter(req.lang)
        result = f.filter(req.text, req.role)

        logger.info(
            "[traceId=%s] filter result: should_embed=%s score=%d reason=%s",
            trace_id, result.should_embed, result.score, result.reason
        )

        return FilterMessageResp(**result.to_dict())
    except Exception as e:
        logger.error("[traceId=%s] filter error: %s", trace_id, str(e))
        # 出错时保守处理：允许 embedding
        return FilterMessageResp(
            should_embed=True,
            score=0,
            coherence=0.5,
            reason=f"error:{str(e)}"
        )


@router.post("/qualify/batch", response_model=BatchFilterResp)
def qualify_batch_messages(req: BatchFilterReq, request: Request):
    """
    批量判断消息是否值得做 embedding
    """
    trace_id = request.headers.get("x-trace-id") or "-"
    logger.debug("[traceId=%s] /embed/qualify/batch count=%d", trace_id, len(req.messages))

    results = []
    for msg in req.messages:
        try:
            f = get_filter(msg.lang)
            result = f.filter(msg.text, msg.role)
            results.append(FilterMessageResp(**result.to_dict()))
        except Exception as e:
            logger.error("[traceId=%s] batch filter error: %s", trace_id, str(e))
            results.append(FilterMessageResp(
                should_embed=True,
                score=0,
                coherence=0.5,
                reason=f"error:{str(e)}"
            ))

    return BatchFilterResp(results=results)


@router.get("/languages")
def list_languages():
    """
    列出支持的语言
    """
    return {"languages": MessageFilter.available_languages()}
