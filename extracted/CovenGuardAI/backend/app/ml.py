from __future__ import annotations
import numpy as np

def sigmoid(z):
    z = np.clip(z, -25, 25)
    return 1.0 / (1.0 + np.exp(-z))

def predict(w, x):
    return (sigmoid(x @ w[:-1] + w[-1]) >= 0.5).astype(np.int64)

def train_local(x, y, initial_w, epochs=2, lr=0.15):
    """Small binary logistic-regression client trainer; returns updated parameter vector."""
    w = initial_w.copy()
    if len(x) == 0:
        return w
    for _ in range(epochs):
        p = sigmoid(x @ w[:-1] + w[-1])
        err = p - y
        grad = np.concatenate([(x.T @ err) / len(x), [err.mean()]])
        w -= lr * grad
    return w

def metrics(w, x, y):
    pred = predict(w, x)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"accuracy": round(accuracy, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}

def fedavg(updates, weights):
    return np.average(np.stack(updates), axis=0, weights=np.asarray(weights))

def coordinate_median(updates):
    return np.median(np.stack(updates), axis=0)

def trimmed_mean(updates, trim_ratio=0.2):
    arr = np.sort(np.stack(updates), axis=0)
    trim = min(int(len(updates) * trim_ratio), max(0, (len(updates)-1)//2))
    if trim:
        arr = arr[trim:-trim]
    return arr.mean(axis=0)

def inspect_updates(updates):
    matrix = np.stack(updates)
    norms = np.linalg.norm(matrix, axis=1)
    unit = matrix / np.maximum(norms[:, None], 1e-12)
    cosine = unit @ unit.T
    median = np.median(matrix, axis=0)
    deviations = np.linalg.norm(matrix - median, axis=1)
    # Robust z-like scores; robust scale avoids a single outlier dominating.
    med_dev = np.median(deviations)
    mad = np.median(np.abs(deviations - med_dev)) + 1e-9
    robust_z = 0.6745 * (deviations - med_dev) / mad
    mean_similarity = (cosine.sum(axis=1) - 1) / max(1, len(updates)-1)
    # Combine magnitude/deviation and directional disagreement.
    flags = (robust_z > 3.5) | (mean_similarity < 0.0)
    return norms, mean_similarity, deviations, robust_z, flags, cosine
