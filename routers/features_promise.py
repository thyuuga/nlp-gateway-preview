from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import logging

from services.promise_features import extract_promise_candidate_async

router = APIRouter()

# ---- logging ----
logger = logging.getLogger("nlp_gateway.promise")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False


class PromiseCandidateReq(BaseModel):
    text: str = Field(..., min_length=1)
    tz: str = Field(default="Asia/Tokyo")
    assistantText: Optional[str] = None  # 预留，MVP 不强依赖


@router.post("/promise/candidate")
async def promise_candidate(req: PromiseCandidateReq) -> Dict[str, Any]:
    logger.info("/features/promise/candidate text=%s", req.text[:60] if req.text else "")

    result = await extract_promise_candidate_async(text=req.text, tz=req.tz, assistant_text=req.assistantText)

    logger.info("/features/promise/candidate => isCandidate=%s type=%s conf=%s",
                result.get("isCandidate"), result.get("type"), result.get("confidence"))
    return result
