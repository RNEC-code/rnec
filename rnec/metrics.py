from __future__ import annotations
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover
    roc_auc_score = None
    average_precision_score = None


def as_window_labels(y_window_tensor_or_array) -> np.ndarray:
    y = np.asarray(y_window_tensor_or_array)
    if y.ndim >= 2:
        y = y.reshape(y.shape[0], -1).max(axis=1)
    return (y > 0).astype(np.int64).reshape(-1)


def _clean_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros_like(scores, dtype=np.float64)
    max_f = float(np.max(scores[finite]))
    min_f = float(np.min(scores[finite]))
    return np.nan_to_num(scores, nan=min_f, posinf=max_f, neginf=min_f)


def prf(y_true: np.ndarray, y_pred: np.ndarray):
    """Fast binary precision/recall/F1 without calling sklearn in a threshold loop."""
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2.0 * p * r / max(1e-12, p + r)
    return float(p), float(r), float(f1)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, num_thresholds: int = 201):
    """
    Exact vectorized best-F1 threshold search.

    The previous implementation scanned quantile thresholds and called sklearn for
    every threshold. On long all-series runs this could become a bottleneck and was
    easy to interrupt inside sklearn. This version sorts scores once and computes
    TP/FP/FN cumulatively over all unique score thresholds.
    """
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    scores = _clean_scores(scores)
    if scores.size == 0 or y_true.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError(f"y_true and scores length mismatch: {y_true.shape[0]} vs {scores.shape[0]}")

    y_bin = (y_true > 0).astype(np.int64)
    total_pos = int(y_bin.sum())
    if total_pos == 0:
        # No positive validation labels. Pick the largest threshold to predict as few positives as possible.
        thr = float(np.max(scores))
        pred = (scores >= thr).astype(np.int64)
        p, r, f1 = prf(y_bin, pred)
        return p, r, f1, thr

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_bin[order]

    # Positions where a score group ends. Thresholding at that score includes all rows up to group end.
    group_end = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    tp = np.cumsum(y_sorted, dtype=np.int64)[group_end].astype(np.float64)
    pred_pos = (group_end + 1).astype(np.float64)
    fp = pred_pos - tp
    fn = float(total_pos) - tp

    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    f1 = 2.0 * precision * recall / np.maximum(1e-12, precision + recall)

    # Tie-breaker: choose the highest threshold among equal F1, which is more conservative.
    best_idx = int(np.argmax(f1))
    thr = float(s_sorted[group_end[best_idx]])
    return float(precision[best_idx]), float(recall[best_idx]), float(f1[best_idx]), thr


def safe_auroc(y_true: np.ndarray, scores: np.ndarray):
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    scores = _clean_scores(scores)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if roc_auc_score is None:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_aupr(y_true: np.ndarray, scores: np.ndarray):
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    scores = _clean_scores(scores)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if average_precision_score is None:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def evaluate_with_val_threshold(y_val, s_val, y_test, s_test, num_thresholds: int = 201):
    p_val, r_val, f1_val, thr = best_f1_threshold(y_val, s_val, num_thresholds=num_thresholds)
    s_test = _clean_scores(s_test)
    pred = (s_test >= thr).astype(np.int64)
    p, r, f1 = prf(y_test, pred)
    p_or, r_or, f1_or, thr_or = best_f1_threshold(y_test, s_test, num_thresholds=num_thresholds)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "auroc": safe_auroc(y_test, s_test),
        "aupr": safe_aupr(y_test, s_test),
        "threshold_val": float(thr),
        "val_precision": p_val,
        "val_recall": r_val,
        "val_f1": f1_val,
        "precision_oracle": p_or,
        "recall_oracle": r_or,
        "f1_oracle": f1_or,
        "threshold_oracle": float(thr_or),
    }
