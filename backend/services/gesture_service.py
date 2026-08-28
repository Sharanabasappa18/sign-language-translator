"""Real-time gesture recognition service using MediaPipe and ML classifier."""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np

from config import (
    GESTURE_COOLDOWN_FRAMES,
    MAX_HANDS,
    MIN_CONFIDENCE,
    MODEL_PATH,
    STABLE_FRAMES_REQUIRED,
)
from ml_model.classifier import GestureClassifier
from utils.gesture_utils import GestureStabilizer
from utils.hand_detector import HandDetector
from utils.language_maps import translate_gesture


class GestureRecognitionService:
    def __init__(self) -> None:
        self._detector = HandDetector(max_hands=MAX_HANDS)
        self._classifier = GestureClassifier(MODEL_PATH)
        self._stabilizer = GestureStabilizer(
            stable_frames=STABLE_FRAMES_REQUIRED,
            min_confidence=MIN_CONFIDENCE,
            cooldown_frames=GESTURE_COOLDOWN_FRAMES,
        )
        self.sentence_buffer: list[str] = []

    def reset(self) -> None:
        self._stabilizer.reset()
        self.sentence_buffer.clear()

    def clear_text(self) -> None:
        self.sentence_buffer.clear()

    def decode_frame(self, frame_b64: str) -> Optional[np.ndarray]:
        try:
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]
            data = base64.b64decode(frame_b64)
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def process_frame(self, frame: np.ndarray, language: str = "en") -> Dict[str, Any]:
        frame = cv2.flip(frame, 1)
        landmarks, result = self._detector.process(frame)

        current_gesture = None
        current_confidence = 0.0
        preview_text = ""
        accepted = False

        if landmarks:
            frame = self._detector.draw(frame, result)
            label, confidence = self._classifier.predict(landmarks)
            current_gesture = label
            current_confidence = confidence

            stabilized = self._stabilizer.update(label, confidence)
            if stabilized and stabilized.is_new:
                word = translate_gesture(stabilized.gesture, language)
                self.sentence_buffer.append(word)
                preview_text = word
                accepted = True
            elif label:
                preview_text = translate_gesture(label, language)

        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "annotated_frame": annotated_b64,
            "current_gesture": current_gesture,
            "confidence": round(current_confidence, 3),
            "preview_text": preview_text,
            "accepted_gesture": accepted,
            "translated_text": " ".join(self.sentence_buffer),
            "model_loaded": self._classifier.is_trained,
        }
