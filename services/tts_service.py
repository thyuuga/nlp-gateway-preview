# nlp-gateway/services/tts_service.py
#
# OpenAI gpt-4o-mini-tts 文字转语音

import os
import re
import logging
from openai import OpenAI

logger = logging.getLogger("nlp_gateway.tts_service")

VOICE_STYLE = """
Speak in a calm, gentle tone.
Pace: slightly slower than normal (0.9x).
Pause naturally between sentences.
Avoid: robotic reading, overly energetic, or exaggerated sweetness.
""".strip()

VALID_VOICES = {"alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"}

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client


def synthesize(text: str, voice: str = "nova") -> bytes:
    """
    将文本转为 MP3 音频 bytes。

    :param text: 要转换的文字
    :param voice: 声音类型 (默认 nova)
    :return: MP3 音频的 bytes
    """
    if voice not in VALID_VOICES:
        voice = "nova"

    # 去掉舞台指示：（轻轻揉了揉眼睛）、(停顿) 等
    clean = re.sub(r"[（(][^）)]*[）)]", "", text).strip()
    # 多余空格合并
    clean = re.sub(r" {2,}", " ", clean)
    if not clean:
        clean = text  # fallback: 全是括号内容时保留原文

    client = _get_client()

    logger.info("TTS synthesize voice=%s text=%s", voice, clean[:60])

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=clean,
        instructions=VOICE_STYLE,
    )

    audio_bytes = response.read()
    logger.info("TTS done size=%.1fKB", len(audio_bytes) / 1024)
    return audio_bytes
