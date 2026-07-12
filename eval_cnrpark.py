"""Evaluate the fill-ratio classifier on the real CNRPark dataset.

CNRPark (Amato et al., http://cnrpark.it) contains 150x150 patches of
individual parking spots cropped from two real camera views (A and B),
each labeled free or busy.

Protocol: the decision threshold is tuned on camera A only, then accuracy,
precision, recall and F1 are reported on camera B (held out), so the test
metric is not fitted to the test data.

Usage:
    python eval_cnrpark.py --data data/cnrpark
"""

import argparse
import os

import cv2
import numpy as np

from detector import preprocess


def collect(root, camera):
    """Yield (path, label) for one camera; label 1 = busy, 0 = free."""
    items = []
    for label_name, label in (("free", 0), ("busy", 1)):
        base = os.path.join(root, camera, label_name)
        if not os.path.isdir(base):
            raise SystemExit(f"Missing folder: {base}")
        for name in os.listdir(base):
            if name.lower().endswith((".jpg", ".png", ".jpeg")):
                items.append((os.path.join(base, name), label))
    return items


def fill_ratios(items):
    """Compute the adaptive-threshold fill ratio for every patch."""
    ratios, labels = [], []
    for path, label in items:
        img = cv2.imread(path)
        if img is None:
            continue
        thresh = preprocess(img)
        ratios.append(cv2.countNonZero(thresh) / thresh.size)
        labels.append(label)
    return np.array(ratios), np.array(labels)


def best_threshold(ratios, labels):
    """Pick the threshold that maximizes accuracy on the tuning set."""
    candidates = np.linspace(0.05, 0.9, 341)
    accs = [( (ratios > t).astype(int) == labels).mean() for t in candidates]
    i = int(np.argmax(accs))
    return candidates[i], accs[i]


def report(name, ratios, labels, t):
    pred = (ratios > t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\n{name}  (n={len(labels)}, threshold={t:.3f})")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (busy predictions that were busy)")
    print(f"  Recall   : {rec:.4f}  (busy spots that were caught)")
    print(f"  F1       : {f1:.4f}")
    print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cnrpark", help="CNRPark root (contains A/ and B/)")
    args = ap.parse_args()

    print("Loading camera A (tuning set)...")
    ra, la = fill_ratios(collect(args.data, "A"))
    print(f"  {len(la)} patches ({int(la.sum())} busy, {int((la == 0).sum())} free)")

    print("Loading camera B (held-out test set)...")
    rb, lb = fill_ratios(collect(args.data, "B"))
    print(f"  {len(lb)} patches ({int(lb.sum())} busy, {int((lb == 0).sum())} free)")

    t, acc_a = best_threshold(ra, la)
    print(f"\nThreshold tuned on camera A: {t:.3f} (accuracy on A: {acc_a:.4f})")

    report("Camera A (tuning set)", ra, la, t)
    report("Camera B (HELD-OUT TEST)", rb, lb, t)


if __name__ == "__main__":
    main()
