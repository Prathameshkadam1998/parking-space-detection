"""Core parking-spot occupancy detection.

Approach: convert the frame to grayscale, blur, apply adaptive thresholding
so edges/texture (cars) become white pixels, then count white pixels inside
each marked parking spot. A mostly-empty spot has little texture, so its
white-pixel count stays low; a parked car pushes the count above a threshold.
"""

import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Spot:
    x: int
    y: int
    w: int
    h: int
    occupied: bool = False
    fill_ratio: float = 0.0  # fraction of thresholded pixels inside the spot


def load_spots(path):
    """Load parking spot rectangles from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return [Spot(**{k: s[k] for k in ("x", "y", "w", "h")}) for s in data["spots"]]


def save_spots(path, rects):
    """Save a list of (x, y, w, h) rectangles to a JSON file."""
    data = {"spots": [{"x": x, "y": y, "w": w, "h": h} for x, y, w, h in rects]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def preprocess(frame):
    """Turn a BGR frame into a binary texture map used for occupancy checks."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 16,
    )
    thresh = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh


def update_spots(frame, spots, fill_threshold=0.22):
    """Classify each spot as occupied/free. Returns number of free spots."""
    thresh = preprocess(frame)
    free = 0
    for spot in spots:
        roi = thresh[spot.y:spot.y + spot.h, spot.x:spot.x + spot.w]
        area = max(spot.w * spot.h, 1)
        spot.fill_ratio = cv2.countNonZero(roi) / area
        spot.occupied = spot.fill_ratio > fill_threshold
        if not spot.occupied:
            free += 1
    return free


def draw_overlay(frame, spots, free_count):
    """Draw colored spot rectangles and a free-count banner onto the frame."""
    for spot in spots:
        color = (0, 0, 255) if spot.occupied else (0, 220, 0)
        cv2.rectangle(frame, (spot.x, spot.y),
                      (spot.x + spot.w, spot.y + spot.h), color, 2)
        cv2.putText(frame, f"{spot.fill_ratio:.2f}",
                    (spot.x + 3, spot.y + spot.h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    banner = f"Free: {free_count}/{len(spots)}"
    cv2.rectangle(frame, (10, 10), (240, 55), (40, 40, 40), -1)
    cv2.putText(frame, banner, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame
