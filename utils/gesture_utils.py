"""Hand landmark feature extraction and gesture stabilization."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple

import numpy as np


def normalize_landmarks(landmarks: Iterable) -> np.ndarray:
    """Convert MediaPipe landmarks to a wrist-relative, scale-invariant vector."""
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = points[0].copy()
    points -= wrist

    # Scale by palm size (wrist to middle MCP)
    scale = np.linalg.norm(points[9]) + 1e-6
    points /= scale
    return points.flatten()


def landmarks_to_features(landmarks: Iterable) -> np.ndarray:
    """63-dim feature vector from 21 hand landmarks."""
    return normalize_landmarks(landmarks)


def finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """Heuristic: finger is extended if tip is farther from wrist than PIP joint."""
    wrist = landmarks[0]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    tip_dist = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
    pip_dist = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
    return tip_dist > pip_dist * 1.05


def rule_based_gesture(landmarks) -> Optional[str]:
    """
    Fallback heuristic classifier for common demo gestures.
    Useful before a trained model exists or when confidence is low.
    """
    thumb = finger_extended(landmarks, 4, 3)
    index = finger_extended(landmarks, 8, 6)
    middle = finger_extended(landmarks, 12, 10)
    ring = finger_extended(landmarks, 16, 14)
    pinky = finger_extended(landmarks, 20, 18)

    extended = [thumb, index, middle, ring, pinky]

    if all(extended):
        return "hello"
    if thumb and not any([index, middle, ring, pinky]):
        return "yes"
    if not thumb and not any([index, middle, ring, pinky]):
        return "no"
    if index and middle and not ring and not pinky and not thumb:
        return "two"
    if index and not middle and not ring and not pinky and not thumb:
        return "one"
    if index and middle and ring and not pinky and not thumb:
        return "three"
    if index and pinky and not middle and not ring:
        return "love"
    if not any(extended):
        return "please"
    if index and middle and ring and pinky and not thumb:
        return "thank_you"
    return None


@dataclass
class StabilizedGesture:
    gesture: str
    confidence: float
    is_new: bool


class GestureStabilizer:
    """
    Accept gestures only when they remain stable across multiple frames,
    reducing accidental triggers from casual hand movement.
    """

    def __init__(
        self,
        stable_frames: int = 12,
        min_confidence: float = 0.72,
        cooldown_frames: int = 20,
    ) -> None:
        self.stable_frames = stable_frames
        self.min_confidence = min_confidence
        self.cooldown_frames = cooldown_frames
        self._history: Deque[Tuple[str, float]] = deque(maxlen=stable_frames)
        self._cooldown = 0

    def reset(self) -> None:
        self._history.clear()
        self._cooldown = 0

    def update(self, gesture: Optional[str], confidence: float) -> Optional[StabilizedGesture]:
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if not gesture or confidence < self.min_confidence:
            self._history.clear()
            return None

        self._history.append((gesture, confidence))

        if len(self._history) < self.stable_frames:
            return None

        labels = [g for g, _ in self._history]
        counts = Counter(labels)
        top_gesture, count = counts.most_common(1)[0]

        if count < self.stable_frames:
            return None

        avg_conf = float(np.mean([c for g, c in self._history if g == top_gesture]))
        self._history.clear()
        self._cooldown = self.cooldown_frames
        return StabilizedGesture(gesture=top_gesture, confidence=avg_conf, is_new=True)
