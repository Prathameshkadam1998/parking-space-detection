"""Generate a synthetic top-down parking lot video for testing the detector.

Creates data/parking.mp4 (two rows of spots, cars arriving and leaving over
time) and data/spots.json with the matching spot rectangles, so the project
runs end-to-end without any real footage.

Usage:
    python generate_demo.py
"""

import os
import random

import cv2
import numpy as np

from detector import save_spots

W, H = 960, 540
FPS = 25
SECONDS = 12
SPOT_W, SPOT_H = 80, 130
COLS = 10
ROW_Y = [70, 340]  # top y of each parking row
LOT_X0 = 60

CAR_COLORS = [
    (60, 60, 200), (200, 120, 60), (70, 160, 70), (50, 50, 50),
    (200, 200, 200), (30, 140, 220), (140, 60, 140), (90, 90, 160),
]


def make_background():
    """Asphalt with mild texture plus painted spot lines."""
    rng = np.random.default_rng(7)
    base = rng.normal(105, 6, (H, W)).clip(80, 130).astype(np.uint8)
    bg = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for row_y in ROW_Y:
        for c in range(COLS + 1):
            x = LOT_X0 + c * SPOT_W
            cv2.line(bg, (x, row_y), (x, row_y + SPOT_H), (235, 235, 235), 3)
        cv2.line(bg, (LOT_X0, row_y), (LOT_X0 + COLS * SPOT_W, row_y),
                 (235, 235, 235), 3)
    return bg


def draw_car(img, x, y, w, h, color):
    """A simple top-down car: rounded body, windshield, roof highlight."""
    m = 8  # margin inside the spot
    body = (x + m, y + m + 6, x + w - m, y + h - m - 6)
    cv2.rectangle(img, body[:2], body[2:], color, -1)
    cv2.rectangle(img, body[:2], body[2:], tuple(int(c * 0.6) for c in color), 2)
    # windshield and rear window
    cv2.rectangle(img, (x + m + 6, y + m + 18), (x + w - m - 6, y + m + 34),
                  (40, 30, 30), -1)
    cv2.rectangle(img, (x + m + 6, y + h - m - 34), (x + w - m - 6, y + h - m - 18),
                  (40, 30, 30), -1)
    # roof highlight
    cv2.rectangle(img, (x + m + 10, y + m + 42), (x + w - m - 10, y + h - m - 42),
                  tuple(min(255, int(c * 1.25)) for c in color), -1)


def main():
    os.makedirs("data", exist_ok=True)

    rects = []
    for row_y in ROW_Y:
        for c in range(COLS):
            rects.append((LOT_X0 + c * SPOT_W + 4, row_y + 4,
                          SPOT_W - 8, SPOT_H - 8))
    save_spots("data/spots.json", rects)

    random.seed(42)
    total = len(rects)
    occupied = set(random.sample(range(total), 12))

    # schedule of (frame, spot_index, arrive?) events on initially valid spots
    frames_total = FPS * SECONDS
    free_now = [i for i in range(total) if i not in occupied]
    parked_now = list(occupied)
    events = []
    for t in range(1, 7):
        frame_at = t * frames_total // 7
        if t % 2:  # a car leaves
            idx = parked_now.pop(random.randrange(len(parked_now)))
            events.append((frame_at, idx, False))
            free_now.append(idx)
        else:      # a car arrives
            idx = free_now.pop(random.randrange(len(free_now)))
            events.append((frame_at, idx, True))
            parked_now.append(idx)

    bg = make_background()
    writer = cv2.VideoWriter("data/parking.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    colors = {i: random.choice(CAR_COLORS) for i in range(total)}

    state = set(occupied)
    for f in range(frames_total):
        for frame_at, idx, arrive in events:
            if f == frame_at:
                (state.add if arrive else state.discard)(idx)
        img = bg.copy()
        # per-frame sensor noise so frames aren't identical
        noise = np.random.default_rng(f).integers(-4, 5, (H, W, 1), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for i in state:
            x, y, w, h = rects[i]
            draw_car(img, x, y, w, h, colors[i])
        writer.write(img)
    writer.release()

    print(f"Wrote data/parking.mp4 ({frames_total} frames, {total} spots, "
          f"{len(occupied)} initially occupied) and data/spots.json")


if __name__ == "__main__":
    main()
