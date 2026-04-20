from fastapi import APIRouter, Request
from pydantic import BaseModel
import os
import logging

from services.judge_logic import judge_acceptance, JudgeResp

router = APIRouter()

# ---- logging (minimal) ----
logger = logging.getLogger("nlp_gateway.judge")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False

class JudgeReq(BaseModel):
    user_text: str
    assistant_text: str
    lang: str | None = None

@router.post("/accept", response_model=JudgeResp)
def judge(req: JudgeReq, request: Request):
    trace_id = request.headers.get("x-trace-id") or "-"
    logger.debug("[traceId=%s] /judge/accept", trace_id)
    return judge_acceptance(req.user_text, req.assistant_text, trace_id, logger=logger)
