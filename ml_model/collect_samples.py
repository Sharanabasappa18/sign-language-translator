"""
Collect hand landmark samples from webcam for training.

Usage:
  python -m ml_model.collect_samples --gesture hello --count 40
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from config import DATASET_DIR
from utils.gesture_utils import landmarks_to_features
from utils.hand_detector import HandDetector


def collect(gesture: str, count: int, output_dir: Path = DATASET_DIR) -> None:
    gesture_dir = output_dir / gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)

    detector = HandDetector(max_hands=1)
    cap = cv2.VideoCapture(0)

    saved = 0
    print(f"Collecting '{gesture}' — hold the sign steady. Press 'q' to quit early.")

    try:
        while saved < count:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            landmarks, result = detector.process(frame)
            frame = detector.draw(frame, result)

            if landmarks:
                features = landmarks_to_features(landmarks)
                filename = gesture_dir / f"{gesture}_{int(time.time() * 1000)}_{saved}.npy"
                np.save(filename, features)
                saved += 1
                cv2.putText(
                    frame,
                    f"Saved {saved}/{count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Collect Samples", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} samples to {gesture_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect gesture training samples")
    parser.add_argument("--gesture", required=True, help="Gesture label folder name")
    parser.add_argument("--count", type=int, default=40, help="Number of samples")
    parser.add_argument("--output", type=Path, default=DATASET_DIR)
    args = parser.parse_args()
    collect(args.gesture, args.count, args.output)


if __name__ == "__main__":
    main()
