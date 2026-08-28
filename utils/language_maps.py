"""Gesture-to-text mappings for supported languages."""

from typing import Dict

# Extend this dictionary to add new gestures or languages.
GESTURE_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "hello": {
        "en": "Hello",
        "kn": "ನಮಸ್ಕಾರ",
        "hi": "नमस्ते",
        "ta": "வணக்கம்",
    },
    "yes": {
        "en": "Yes",
        "kn": "ಹೌದು",
        "hi": "हाँ",
        "ta": "ஆம்",
    },
    "no": {
        "en": "No",
        "kn": "ಇಲ್ಲ",
        "hi": "नहीं",
        "ta": "இல்லை",
    },
    "thank_you": {
        "en": "Thank you",
        "kn": "ಧನ್ಯವಾದ",
        "hi": "धन्यवाद",
        "ta": "நன்றி",
    },
    "please": {
        "en": "Please",
        "kn": "ದಯವಿಟ್ಟು",
        "hi": "कृपया",
        "ta": "தயவுசெய்து",
    },
    "love": {
        "en": "I love you",
        "kn": "ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ",
        "hi": "मैं तुमसे प्यार करता हूँ",
        "ta": "நான் உன்னை காதலிக்கிறேன்",
    },
    "one": {
        "en": "One",
        "kn": "ಒಂದು",
        "hi": "एक",
        "ta": "ஒன்று",
    },
    "two": {
        "en": "Two",
        "kn": "ಎರಡು",
        "hi": "दो",
        "ta": "இரண்டு",
    },
    "three": {
        "en": "Three",
        "kn": "ಮೂರು",
        "hi": "तीन",
        "ta": "மூன்று",
    },
    "a": {"en": "A", "kn": "ಎ", "hi": "ए", "ta": "அ"},
    "b": {"en": "B", "kn": "ಬಿ", "hi": "ब", "ta": "ப"},
    "c": {"en": "C", "kn": "ಸಿ", "hi": "स", "ta": "ச"},
}


def translate_gesture(gesture: str, language: str) -> str:
    """Return localized text for a recognized gesture label."""
    lang = language if language in ("en", "kn", "hi", "ta") else "en"
    entry = GESTURE_TRANSLATIONS.get(gesture, {})
    return entry.get(lang, entry.get("en", gesture.replace("_", " ").title()))


def list_gestures() -> list[str]:
    return sorted(GESTURE_TRANSLATIONS.keys())
