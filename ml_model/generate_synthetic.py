"""
Generate synthetic landmark samples for bootstrapping the classifier.

Creates perturbed feature vectors around rule-based gesture prototypes
so the app works out of the box before real data is collected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import DATASET_DIR, MODEL_PATH
from ml_model.train import train
from utils.language_maps import list_gestures


def _base_vector(pattern: str) -> np.ndarray:
    """Build a rough 63-dim prototype from finger extension pattern (T,I,M,R,P)."""
    vec = np.zeros(63, dtype=np.float32)
    fingers = {"T": 0, "I": 1, "M": 2, "R": 3, "P": 4}
    for i, ch in enumerate(pattern):
        if ch == "1" and i < 5:
            base = fingers[["T", "I", "M", "R", "P"][i]] * 3
            vec[base : base + 3] = [0.2, -0.4, 0.0]
        elif ch == "0" and i < 5:
            base = fingers[["T", "I", "M", "R", "P"][i]] * 3
            vec[base : base + 3] = [0.05, -0.1, 0.0]
    # palm anchor
    vec[9:12] = [0.0, -0.2, 0.0]
    vec[0:3] = [0.0, 0.0, 0.0]
    return vec


PROTOTYPES = {
    "hello": "11111",
    "yes": "10000",
    "no": "00000",
    "one": "01000",
    "two": "01100",
    "three": "01110",
    "love": "01001",
    "please": "00000",
    "thank_you": "01111",
    "a": "10000",
    "b": "01111",
    "c": "01111",
}


def generate(
    output_dir: Path = DATASET_DIR,
    samples_per_gesture: int = 120,
    noise: float = 0.04,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for gesture in list_gestures():
        pattern = PROTOTYPES.get(gesture, "00000")
        proto = _base_vector(pattern)
        gesture_dir = output_dir / gesture
        gesture_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_gesture):
            sample = proto + np.random.normal(0, noise, size=63).astype(np.float32)
            np.save(gesture_dir / f"synthetic_{i:04d}.npy", sample)

    print(f"Generated synthetic dataset in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--output", type=Path, default=DATASET_DIR)
    parser.add_argument("--train", action="store_true", help="Train model after generation")
    args = parser.parse_args()

    generate(args.output, args.samples)
    if args.train:
        train(args.output, MODEL_PATH)


if __name__ == "__main__":
    main()
