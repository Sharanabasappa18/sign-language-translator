"""Optional server-side text-to-speech via gTTS."""

from __future__ import annotations

import io
from typing import Optional

from config import LANGUAGES

GTTS_LANG_MAP = {
    "en": "en",
    "kn": "kn",
    "hi": "hi",
    "ta": "ta",
}


def synthesize_speech(text: str, language: str = "en") -> Optional[bytes]:
    """
    Generate MP3 audio bytes for the given text.
    Returns None if gTTS is unavailable or synthesis fails.
    """
    try:
        from gtts import gTTS
    except ImportError:
        return None

    lang_code = GTTS_LANG_MAP.get(language, "en")
    try:
        tts = gTTS(text=text, lang=lang_code)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        return buffer.getvalue()
    except Exception:
        return None


def get_browser_tts_code(language: str) -> str:
    return LANGUAGES.get(language, LANGUAGES["en"])["tts_code"]
