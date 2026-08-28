"""Application configuration."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "ml_model" / "gesture_classifier.pkl"
DATASET_DIR = ROOT_DIR / "training_dataset"
FRONTEND_DIR = ROOT_DIR / "frontend"

# Gesture recognition thresholds
MIN_CONFIDENCE = 0.72
STABLE_FRAMES_REQUIRED = 12
GESTURE_COOLDOWN_FRAMES = 20
MAX_HANDS = 1

# Supported output languages
LANGUAGES = {
    "en": {"name": "English", "tts_code": "en-US"},
    "kn": {"name": "Kannada", "tts_code": "kn-IN"},
    "hi": {"name": "Hindi", "tts_code": "hi-IN"},
    "ta": {"name": "Tamil", "tts_code": "ta-IN"},
}

DEFAULT_LANGUAGE = "en"
