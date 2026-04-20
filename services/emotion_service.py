# nlp-gateway/services/emotion_service.py
#
# Emotion 分类模块 - 基于 MacBERT 的单句情绪判断
#
# 6 类: 平常 / 开心 / 伤心 / 生气 / 得意 / 害羞
#
# 懒加载模型，首次调用时初始化。

import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("nlp_gateway.emotion")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False

# ---- 阈值 ----
CONF_HIGH = 0.60
CONF_LOW = 0.45

# ---- 中文 → 英文标签映射 ----
ZH_TO_EN = {
    "平常": "neutral",
    "开心": "happy",
    "伤心": "sad",
    "生气": "angry",
    "得意": "proud",
    "害羞": "shy",
}

# ---- Lazy Load ----
_emotion_model = None
_emotion_tokenizer = None


def _get_emotion_model():
    """
    Lazy load emotion classifier model.
    Returns (model, tokenizer) or (None, None) on failure.
    """
    global _emotion_model, _emotion_tokenizer
    if _emotion_model is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_model")
            _emotion_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            _emotion_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            _emotion_model.eval()
            logger.info("Emotion classifier loaded from %s", model_dir)
        except Exception as e:
            logger.warning("Failed to load emotion classifier: %s", e)
            _emotion_model = "FAILED"
    if _emotion_model == "FAILED":
        return None, None
    return _emotion_model, _emotion_tokenizer


def predict_emotion(text: str) -> Optional[Dict[str, Any]]:
    """
    对单句文本进行情绪分类。

    Returns:
        {
            "emotion": str,          # 最终判定的情绪标签
            "confidence": float,     # 最高概率
            "candidates": [          # 全部分类及概率（降序）
                {"emotion": str, "prob": float}, ...
            ]
        }
        或 None（模型不可用时）
    """
    model, tokenizer = _get_emotion_model()
    if model is None:
        return None

    try:
        import torch

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64,
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        id2label = model.config.id2label

        candidates = []
        for i, p in enumerate(probs):
            zh_label = id2label.get(i, id2label.get(str(i), f"unknown_{i}"))
            en_label = ZH_TO_EN.get(zh_label, zh_label)
            candidates.append({
                "emotion": en_label,
                "label_zh": zh_label,
                "prob": round(float(p), 4),
            })

        candidates.sort(key=lambda x: x["prob"], reverse=True)

        top = candidates[0]

        # 阈值策略：低于 CONF_LOW 回退为 "neutral"
        if top["prob"] >= CONF_HIGH:
            final_emotion = top["emotion"]
        elif top["prob"] >= CONF_LOW:
            final_emotion = top["emotion"]
        else:
            final_emotion = "neutral"

        return {
            "emotion": final_emotion,
            "label_zh": top["label_zh"] if top["prob"] >= CONF_LOW else "平常",
            "confidence": top["prob"],
            "candidates": candidates,
        }

    except Exception as e:
        logger.warning("predict_emotion failed: %s", e)
        return None
