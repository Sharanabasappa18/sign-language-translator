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


def _make_hand_prototype(gesture: str) -> np.ndarray:
    """Build a distinct, normalized 63-dim feature vector for 21 hand landmarks."""
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist anchor
    pts[0] = [0.0, 0.0, 0.0]

    # MCP positions
    pts[1] = [-0.15, -0.15, 0.0]  # Thumb CMC
    pts[2] = [-0.28, -0.30, 0.0]  # Thumb MCP
    pts[5] = [-0.20, -0.55, 0.0]  # Index MCP
    pts[9] = [0.00, -0.60, 0.0]   # Middle MCP
    pts[13] = [0.20, -0.55, 0.0]  # Ring MCP
    pts[17] = [0.35, -0.48, 0.0]  # Pinky MCP

    def extend_finger(mcp_idx, pip_idx, dip_idx, tip_idx, x_off, y_len):
        mcp = pts[mcp_idx]
        pts[pip_idx] = [mcp[0] + x_off * 0.4, mcp[1] - y_len * 0.4, 0.0]
        pts[dip_idx] = [mcp[0] + x_off * 0.7, mcp[1] - y_len * 0.7, 0.0]
        pts[tip_idx] = [mcp[0] + x_off, mcp[1] - y_len, 0.0]

    def fold_finger(mcp_idx, pip_idx, dip_idx, tip_idx):
        mcp = pts[mcp_idx]
        pts[pip_idx] = [mcp[0], mcp[1] - 0.15, -0.1]
        pts[dip_idx] = [mcp[0], mcp[1] - 0.05, -0.15]
        pts[tip_idx] = [mcp[0], mcp[1] + 0.05, -0.05]

    def extend_thumb_up():
        pts[3] = [-0.35, -0.50, 0.0]
        pts[4] = [-0.40, -0.75, 0.0]

    def extend_thumb_out():
        pts[3] = [-0.45, -0.35, 0.0]
        pts[4] = [-0.60, -0.45, 0.0]

    def fold_thumb_across():
        pts[3] = [-0.10, -0.30, -0.1]
        pts[4] = [0.05, -0.35, -0.15]

    def fold_thumb_fist():
        pts[3] = [-0.18, -0.35, -0.05]
        pts[4] = [-0.10, -0.45, -0.08]

    if gesture == "hello":
        extend_thumb_out()
        extend_finger(5, 6, 7, 8, -0.05, 0.50)
        extend_finger(9, 10, 11, 12, 0.00, 0.55)
        extend_finger(13, 14, 15, 16, 0.05, 0.50)
        extend_finger(17, 18, 19, 20, 0.10, 0.42)
    elif gesture == "yes":
        extend_thumb_up()
        fold_finger(5, 6, 7, 8)
        fold_finger(9, 10, 11, 12)
        fold_finger(13, 14, 15, 16)
        fold_finger(17, 18, 19, 20)
    elif gesture == "no":
        fold_thumb_fist()
        fold_finger(5, 6, 7, 8)
        fold_finger(9, 10, 11, 12)
        fold_finger(13, 14, 15, 16)
        fold_finger(17, 18, 19, 20)
    elif gesture == "one":
        fold_thumb_across()
        extend_finger(5, 6, 7, 8, 0.00, 0.55)
        fold_finger(9, 10, 11, 12)
        fold_finger(13, 14, 15, 16)
        fold_finger(17, 18, 19, 20)
    elif gesture == "two":
        fold_thumb_across()
        extend_finger(5, 6, 7, 8, -0.08, 0.52)
        extend_finger(9, 10, 11, 12, 0.08, 0.55)
        fold_finger(13, 14, 15, 16)
        fold_finger(17, 18, 19, 20)
    elif gesture == "three":
        fold_thumb_across()
        extend_finger(5, 6, 7, 8, -0.06, 0.52)
        extend_finger(9, 10, 11, 12, 0.00, 0.55)
        extend_finger(13, 14, 15, 16, 0.06, 0.50)
        fold_finger(17, 18, 19, 20)
    elif gesture == "love":
        extend_thumb_out()
        extend_finger(5, 6, 7, 8, -0.05, 0.52)
        fold_finger(9, 10, 11, 12)
        fold_finger(13, 14, 15, 16)
        extend_finger(17, 18, 19, 20, 0.08, 0.45)
    elif gesture == "thank_you":
        fold_thumb_across()
        extend_finger(5, 6, 7, 8, -0.03, 0.50)
        extend_finger(9, 10, 11, 12, 0.00, 0.55)
        extend_finger(13, 14, 15, 16, 0.03, 0.50)
        extend_finger(17, 18, 19, 20, 0.06, 0.42)
    elif gesture == "please":
        fold_thumb_fist()
        extend_finger(5, 6, 7, 8, 0.15, 0.45)
        extend_finger(9, 10, 11, 12, 0.15, 0.48)
        extend_finger(13, 14, 15, 16, 0.15, 0.45)
        extend_finger(17, 18, 19, 20, 0.15, 0.38)
    elif gesture == "a":
        pts[3] = [-0.22, -0.42, 0.0]
        pts[4] = [-0.22, -0.60, 0.0]
        fold_finger(5, 6, 7, 8)
        fold_finger(9, 10, 11, 12)
        fold_finger(13, 14, 15, 16)
        fold_finger(17, 18, 19, 20)
    elif gesture == "b":
        pts[3] = [-0.08, -0.32, -0.15]
        pts[4] = [0.00, -0.38, -0.18]
        extend_finger(5, 6, 7, 8, 0.00, 0.52)
        extend_finger(9, 10, 11, 12, 0.00, 0.56)
        extend_finger(13, 14, 15, 16, 0.00, 0.52)
        extend_finger(17, 18, 19, 20, 0.00, 0.45)
    elif gesture == "c":
        pts[3] = [-0.35, -0.35, 0.1]
        pts[4] = [-0.25, -0.45, 0.2]
        for mcp_idx, pip_idx, dip_idx, tip_idx in [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]:
            mcp = pts[mcp_idx]
            pts[pip_idx] = [mcp[0], mcp[1] - 0.25, 0.1]
            pts[dip_idx] = [mcp[0], mcp[1] - 0.40, 0.2]
            pts[tip_idx] = [mcp[0] + 0.1, mcp[1] - 0.42, 0.25]
    else:
        fold_thumb_fist()
        fold_finger(5, 6, 7, 8)
        fold_finger(9, 10, 11, 12)
        fold_finger(13, 14, 15, 16)
        fold_finger(17, 18, 19, 20)

    # Normalize: wrist-relative, scale invariant
    pts -= pts[0].copy()
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()


def generate(
    output_dir: Path = DATASET_DIR,
    samples_per_gesture: int = 120,
    noise: float = 0.04,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for gesture in list_gestures():
        proto = _make_hand_prototype(gesture)
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
