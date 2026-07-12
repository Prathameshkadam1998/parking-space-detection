"""Run parking space detection on a video, webcam, or single image.

Usage:
    python main.py --video data/parking.mp4 --spots data/spots.json
    python main.py --video 0                              # webcam
    python main.py --video data/parking.mp4 --save out.mp4 --headless

Controls (window mode):
    q / ESC   quit
    space     pause / resume
"""

import argparse

import cv2

from detector import load_spots, update_spots, draw_overlay


def main():
    ap = argparse.ArgumentParser(description="Parking space detection")
    ap.add_argument("--video", required=True, help="video file, image, or camera index")
    ap.add_argument("--spots", default="data/spots.json", help="spots JSON from select_spots.py")
    ap.add_argument("--threshold", type=float, default=0.22,
                    help="fill ratio above which a spot counts as occupied")
    ap.add_argument("--save", default=None, help="write annotated video to this path")
    ap.add_argument("--headless", action="store_true", help="no display window")
    args = ap.parse_args()

    spots = load_spots(args.spots)
    print(f"Loaded {len(spots)} spots from {args.spots}")

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    paused = False
    frame = None
    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                # loop video files; stop for cameras/images
                if isinstance(source, str) and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1 and not args.save:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            free = update_spots(frame, spots, args.threshold)
            draw_overlay(frame, spots, free)
            if writer:
                writer.write(frame)

        if not args.headless:
            cv2.imshow("Parking detection  (q=quit, space=pause)", frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to {args.save}")
    cv2.destroyAllWindows()

    free = sum(1 for s in spots if not s.occupied)
    print(f"Last frame: {free}/{len(spots)} spots free")


if __name__ == "__main__":
    main()
