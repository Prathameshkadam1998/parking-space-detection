"""Interactive tool to mark parking spots on a video frame.

Usage:
    python select_spots.py --video data/parking.mp4 --spots data/spots.json

Controls:
    Left click + drag   draw a new spot rectangle
    Right click         delete the spot under the cursor
    u                   undo last spot
    s                   save and quit
    q / ESC             quit without saving
"""

import argparse
import os

import cv2

from detector import load_spots, save_spots


def main():
    ap = argparse.ArgumentParser(description="Mark parking spots on a frame")
    ap.add_argument("--video", required=True, help="video file or camera index")
    ap.add_argument("--spots", default="data/spots.json", help="output JSON path")
    args = ap.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.video}")

    rects = []
    if os.path.exists(args.spots):
        rects = [(s.x, s.y, s.w, s.h) for s in load_spots(args.spots)]
        print(f"Loaded {len(rects)} existing spots from {args.spots}")

    state = {"drawing": False, "start": None, "current": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x0, y0 = state["start"]
            rx, ry = min(x0, x), min(y0, y)
            rw, rh = abs(x - x0), abs(y - y0)
            if rw > 5 and rh > 5:
                rects.append((rx, ry, rw, rh))
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (rx, ry, rw, rh) in enumerate(rects):
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    rects.pop(i)
                    break

    win = "Select spots  (drag=add, right-click=delete, u=undo, s=save, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = frame.copy()
        for rx, ry, rw, rh in rects:
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
        if state["drawing"] and state["start"] and state["current"]:
            cv2.rectangle(canvas, state["start"], state["current"], (0, 255, 255), 1)
        cv2.putText(canvas, f"Spots: {len(rects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and rects:
            rects.pop()
        elif key == ord("s"):
            os.makedirs(os.path.dirname(args.spots) or ".", exist_ok=True)
            save_spots(args.spots, rects)
            print(f"Saved {len(rects)} spots to {args.spots}")
            break
        elif key in (ord("q"), 27):
            print("Quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
