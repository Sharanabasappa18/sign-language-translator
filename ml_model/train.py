"""
Train the gesture classifier from collected landmark samples.

Dataset layout:
  training_dataset/<gesture_label>/*.npy   # each file is a (63,) feature vector

Collect samples with:
  python -m ml_model.collect_samples --gesture hello --count 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import DATASET_DIR, MODEL_PATH
from ml_model.classifier import GestureClassifier
from utils.language_maps import list_gestures


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for gesture_dir in sorted(dataset_dir.iterdir()):
        if not gesture_dir.is_dir():
            continue
        label = gesture_dir.name
        for sample_file in gesture_dir.glob("*.npy"):
            X.append(np.load(sample_file))
            y.append(label)
    if not X:
        raise FileNotFoundError(
            f"No training samples found under {dataset_dir}. "
            "Run collect_samples.py or generate_synthetic.py first."
        )
    return np.array(X), np.array(y)


def train(dataset_dir: Path = DATASET_DIR, model_path: Path = MODEL_PATH) -> None:
    X, y = load_dataset(dataset_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = GestureClassifier.build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    classifier = GestureClassifier()
    classifier.pipeline = pipeline
    classifier.save(model_path)
    print(f"Model saved to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help="Path to training_dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help="Output model path",
    )
    args = parser.parse_args()
    train(args.dataset, args.output)


if __name__ == "__main__":
    main()
