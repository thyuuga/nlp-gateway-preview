from functools import lru_cache
import spacy

_LANG_MODEL = {
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "en": "en_core_web_sm",
}


@lru_cache(maxsize=8)
def get_nlp(lang: str):
    lang = (lang or "zh").lower()
    model = _LANG_MODEL.get(lang, _LANG_MODEL["zh"])
    try:
        return spacy.load(model)
    except Exception:
        # Fallback: blank pipeline so service still works even if model not installed.
        # Note: blank pipeline has weaker linguistic features; signals will be conservative.
        try:
            return spacy.blank(lang)
        except Exception:
            return spacy.blank("xx")
