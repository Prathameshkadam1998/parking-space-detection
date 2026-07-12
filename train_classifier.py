"""Train a texture-feature classifier on CNRPark and report held-out metrics.

Instead of a single fill-ratio threshold, each spot patch is described by a
small vector of hand-crafted texture/edge features, and a logistic regression
learns the free-vs-busy boundary.

Protocol (cross-camera generalization, matches the CNRPark literature):
    train on camera A  ->  test on camera B (never seen during training)

Usage:
    python train_classifier.py --data data/cnrpark
Saves the trained model to data/spot_classifier.joblib.
"""

import argparse
import os

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from detector import preprocess
from eval_cnrpark import collect


def extract_features(img):
    """Small texture/edge feature vector for one BGR spot patch."""
    img = cv2.resize(img, (96, 96))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = preprocess(img)
    fill_ratio = cv2.countNonZero(thresh) / thresh.size

    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / edges.size

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = hsv[:, :, 1].mean()

    # gradient energy split into a coarse 3x3 grid captures where the
    # texture is (cars fill the center; asphalt texture is uniform)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    cells = [mag[r:r + 32, c:c + 32].mean()
             for r in range(0, 96, 32) for c in range(0, 96, 32)]

    return [fill_ratio, edge_density, lap_var,
            gray.mean(), gray.std(), sat_mean, *cells]


def load_features(items, tag):
    X, y = [], []
    for i, (path, label) in enumerate(items):
        img = cv2.imread(path)
        if img is None:
            continue
        X.append(extract_features(img))
        y.append(label)
        if (i + 1) % 2000 == 0:
            print(f"  {tag}: {i + 1}/{len(items)} patches")
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cnrpark")
    ap.add_argument("--model-out", default="data/spot_classifier.joblib")
    args = ap.parse_args()

    print("Extracting features: camera A (train)...")
    Xa, ya = load_features(collect(args.data, "A"), "A")
    print("Extracting features: camera B (held-out test)...")
    Xb, yb = load_features(collect(args.data, "B"), "B")
    print(f"Train: {len(ya)} patches | Test: {len(yb)} patches | "
          f"{Xa.shape[1]} features")

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
    model.fit(Xa, ya)

    for name, X, y in (("Camera A (train)", Xa, ya),
                       ("Camera B (HELD-OUT TEST)", Xb, yb)):
        pred = model.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        print(f"\n{name}  (n={len(y)})")
        print(f"  Accuracy : {accuracy_score(y, pred):.4f}")
        print(f"  Precision: {precision_score(y, pred):.4f}")
        print(f"  Recall   : {recall_score(y, pred):.4f}")
        print(f"  F1       : {f1_score(y, pred):.4f}")
        print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

    joblib.dump(model, args.model_out)
    print(f"\nSaved model to {args.model_out}")


if __name__ == "__main__":
    main()
