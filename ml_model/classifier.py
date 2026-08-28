"""Gesture classifier using scikit-learn on hand landmark features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.gesture_utils import landmarks_to_features, rule_based_gesture


class GestureClassifier:
    """Wraps a trained model with rule-based fallback."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        if model_path and model_path.exists():
            self.pipeline = joblib.load(model_path)

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def predict(self, landmarks) -> Tuple[Optional[str], float]:
        """Return (gesture_label, confidence)."""
        rule_label = rule_based_gesture(landmarks)
        features = landmarks_to_features(landmarks).reshape(1, -1)

        if self.pipeline is not None:
            proba = self.pipeline.predict_proba(features)[0]
            classes = list(self.pipeline.classes_)
            best_idx = int(np.argmax(proba))
            ml_label = classes[best_idx]
            ml_conf = float(proba[best_idx])

            if ml_conf >= 0.55:
                return ml_label, ml_conf

        if rule_label:
            return rule_label, 0.78

        return None, 0.0

    def save(self, path: Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("No trained pipeline to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @staticmethod
    def build_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
