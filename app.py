# ---- load project root .env ----
from pathlib import Path
from dotenv import load_dotenv
_project_root = Path(__file__).resolve().parent
load_dotenv(_project_root / ".env")

# ---- transformers 5.x compat patch for HanLP 2.1.x ----
# HanLP calls BertTokenizer.encode_plus() which was removed in transformers 5.x
try:
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, 'encode_plus'):
        def _encode_plus(self, text, text_pair=None, **kwargs):
            return self(text, text_pair, **kwargs)
        PreTrainedTokenizerBase.encode_plus = _encode_plus
    if not hasattr(PreTrainedTokenizerBase, 'batch_encode_plus'):
        def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
            return self(batch_text_or_text_pairs, **kwargs)
        PreTrainedTokenizerBase.batch_encode_plus = _batch_encode_plus
except ImportError:
    pass

from fastapi import FastAPI
from routers.judge import router as judge_router
from routers.time_parse import router as time_router
from routers.features_promise import router as promise_router
from routers.hard_write import router as hard_write_router
from routers.embed_qualify import router as embed_qualify_router
from routers.embed_encode import router as embed_encode_router
from routers.emotion import router as emotion_router
from routers.tts import router as tts_router
import os
import logging
from datetime import datetime, timezone
from services.judge_logic import NEG

app = FastAPI(title="nlp-gateway", version="0.1.0")

app.include_router(judge_router, prefix="/judge", tags=["judge"])
app.include_router(time_router, prefix="/features", tags=["features"])
app.include_router(promise_router, prefix="/features", tags=["features"])
app.include_router(hard_write_router, prefix="/hard_write", tags=["hard_write"])
app.include_router(embed_qualify_router, prefix="/embed", tags=["embed"])
app.include_router(embed_encode_router, prefix="/embed", tags=["embed"])
app.include_router(emotion_router, prefix="/emotion", tags=["emotion"])
app.include_router(tts_router, prefix="/tts", tags=["tts"])

# ---- logging ----
logger = logging.getLogger("nlp_gateway")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False

# 关掉 uvicorn 默认的 access log（业务日志已覆盖）
logging.getLogger("uvicorn.access").disabled = True

@app.on_event("startup")
def banner():
    start_ts = datetime.now(timezone.utc).isoformat()
    logger.info("============================================================")
    logger.info("nlp-gateway starting ts=%s", start_ts)
    logger.info("NEG=%d endpoints=/health,/judge/accept,/features/time/parse,/features/promise/candidate,/hard_write/judge,/embed/qualify,/embed/encode,/emotion/analyze,/tts/speak", len(NEG))
    logger.info("LOG_LEVEL=%s", (os.getenv("LOG_LEVEL") or "INFO").upper())
    logger.info("============================================================")

@app.get("/health")
def health():
    return {"ok": True}
