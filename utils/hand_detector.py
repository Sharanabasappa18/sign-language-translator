"""MediaPipe 1.x hand landmarker wrapper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import drawing_utils, hand_landmarker

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_model" / "hand_landmarker.task"


class HandDetector:
    """Detect hand landmarks using MediaPipe Tasks API."""

    def __init__(self, max_hands: int = 1) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing model file: {MODEL_PATH}. "
                "Download hand_landmarker.task from MediaPipe model zoo."
            )

        options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options_module.BaseOptions(
                model_asset_path=str(MODEL_PATH.resolve())
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._detector = hand_landmarker.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> Tuple[Optional[List[Any]], Any]:
        """Return (landmarks_list, detection_result) for the first frame pass."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self._detector.detect_for_video(mp_image, self._timestamp_ms)

        landmarks = None
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
        return landmarks, result

    def draw(self, frame_bgr: np.ndarray, result: Any) -> np.ndarray:
        """Draw landmarks onto a BGR frame."""
        annotated = frame_bgr.copy()
        if not result.hand_landmarks:
            return annotated

        for hand_landmarks in result.hand_landmarks:
            drawing_utils.draw_landmarks(
                annotated,
                hand_landmarks,
                hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2),
            )
        return annotated

    def close(self) -> None:
        self._detector.close()
