from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _robust_scale(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    med = float(np.median(x)) if x.size else 0.0
    if x.size:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = max(float(q75 - q25), eps)
    else:
        iqr = 1.0
    return (x - med) / iqr, med, iqr


def robust_standardize_with_ref(x, med, iqr, eps=1e-8):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return ((x - float(med)) / (float(iqr) + eps)).astype(np.float32)


def _standardize_fit(z):
    z = np.asarray(z, dtype=np.float32)
    mu = z.mean(axis=0, keepdims=True)
    std = z.std(axis=0, keepdims=True) + 1e-6
    return mu.astype(np.float32), std.astype(np.float32)


def _softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _chunked_knn(query, ref, k=20, chunk=2048, exclude_self=False):
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    n = query.shape[0]
    k_eff = int(min(max(1, k), ref.shape[0]))
    all_dist = np.empty((n, k_eff), dtype=np.float32)
    all_idx = np.empty((n, k_eff), dtype=np.int64)
    ref_norm = (ref ** 2).sum(axis=1)[None, :]
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        q = query[a:b]
        q_norm = (q ** 2).sum(axis=1)[:, None]
        d2 = q_norm + ref_norm - 2.0 * (q @ ref.T)
        np.maximum(d2, 0.0, out=d2)
        if exclude_self and query.shape[0] == ref.shape[0]:
            rows = np.arange(a, b)
            rows = rows[rows < d2.shape[1]]
            d2[np.arange(len(rows)), rows] = np.inf
        idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        d = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(d, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        d = np.take_along_axis(d, order, axis=1)
        all_idx[a:b] = idx
        all_dist[a:b] = np.sqrt(d + 1e-12).astype(np.float32)
    return all_dist, all_idx


def weighted_quantile(values, weights=None, q=0.95):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    q = float(np.clip(q, 0.0, 1.0))
    if weights is None:
        return float(np.quantile(values, q))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 1e-12:
        return float(np.quantile(values, q))
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / (np.sum(w) + 1e-12)
    return float(v[np.searchsorted(cdf, q, side="left").clip(0, len(v) - 1)])


@dataclass
class LocalEvidenceCalibrator:
    k: int = 20
    tau_quantile: float = 0.95
    max_ref: int = 20000
    seed: int = 42
    chunk: int = 2048
    eps: float = 1e-6
    softplus_temp: float = 1.0

    def fit(self, z_train: np.ndarray, e_train: np.ndarray):
        z_train = np.asarray(z_train, dtype=np.float32)
        e_train = np.asarray(e_train, dtype=np.float32).reshape(-1)
        self.z_mu, self.z_std = _standardize_fit(z_train)
        z_std = (z_train - self.z_mu) / self.z_std
        n = z_std.shape[0]
        rng = np.random.default_rng(self.seed)
        if n > self.max_ref:
            self.ref_idx = np.sort(rng.choice(n, size=self.max_ref, replace=False))
        else:
            self.ref_idx = np.arange(n, dtype=np.int64)
        self.z_ref = z_std[self.ref_idx].astype(np.float32)
        self.e_ref = e_train[self.ref_idx].astype(np.float32)

        d_ref, _ = _chunked_knn(
            self.z_ref,
            self.z_ref,
            k=min(self.k + 1, len(self.z_ref)),
            chunk=self.chunk,
            exclude_self=True,
        )
        kdist_ref = d_ref.mean(axis=1)
        kd_z, _, _ = _robust_scale(kdist_ref, self.eps)
        er_z, er_med, er_iqr = _robust_scale(self.e_ref, self.eps)
        density_score = 1.0 / (1.0 + np.exp(kd_z))
        err_score = 1.0 / (1.0 + np.exp(er_z))
        self.ref_reliability = np.sqrt(np.clip(density_score * err_score, 0.0, 1.0)).astype(np.float32)
        self.err_med = float(er_med)
        self.err_iqr = float(er_iqr)

        raw = self._raw_components(z_train, e_train, q=self.tau_quantile)
        self.sur_med = float(np.median(raw["surprise_soft_raw"]))
        self.sur_iqr = float(
            np.percentile(raw["surprise_soft_raw"], 75) - np.percentile(raw["surprise_soft_raw"], 25) + self.eps
        )
        self.rare_med = float(np.median(raw["rarity_raw"]))
        self.rare_iqr = float(np.percentile(raw["rarity_raw"], 75) - np.percentile(raw["rarity_raw"], 25) + self.eps)
        return self

    def local_stats(self, z_query: np.ndarray, q: float | None = None):
        z_query = np.asarray(z_query, dtype=np.float32)
        zq = (z_query - self.z_mu) / self.z_std
        d, idx = _chunked_knn(zq, self.z_ref, k=min(self.k, len(self.z_ref)), chunk=self.chunk, exclude_self=False)
        tau = np.empty(zq.shape[0], dtype=np.float32)
        sigma = np.empty(zq.shape[0], dtype=np.float32)
        q_use = self.tau_quantile if q is None else float(q)
        for i in range(zq.shape[0]):
            neigh_e = self.e_ref[idx[i]]
            neigh_w = self.ref_reliability[idx[i]]
            tau[i] = weighted_quantile(neigh_e, neigh_w, q=q_use)
            q75, q25 = np.percentile(neigh_e, [75, 25])
            scale = float(q75 - q25)
            if scale < self.eps:
                scale = self.err_iqr
            sigma[i] = scale + self.eps
        rarity_raw = d.mean(axis=1).astype(np.float32)
        return tau, sigma, rarity_raw

    def _raw_components(self, z_query: np.ndarray, e_query: np.ndarray, q: float | None = None):
        e_query = np.asarray(e_query, dtype=np.float32).reshape(-1)
        tau, sigma, rarity_raw = self.local_stats(z_query, q=q)
        residual_z = ((e_query - tau) / (sigma + self.eps)).astype(np.float32)
        t = max(float(self.softplus_temp), self.eps)
        surprise_soft_raw = (t * _softplus(residual_z / t)).astype(np.float32)
        return {
            "residual_z": residual_z,
            "surprise_soft_raw": surprise_soft_raw,
            "rarity_raw": rarity_raw.astype(np.float32),
            "tau": tau.astype(np.float32),
            "sigma": sigma.astype(np.float32),
        }

    def score(self, z_query: np.ndarray, e_query: np.ndarray, return_components: bool = False):
        e_query = np.asarray(e_query, dtype=np.float32).reshape(-1)
        raw = self._raw_components(z_query, e_query, q=self.tau_quantile)
        surprise = robust_standardize_with_ref(raw["surprise_soft_raw"], self.sur_med, self.sur_iqr, self.eps)
        rarity = robust_standardize_with_ref(raw["rarity_raw"], self.rare_med, self.rare_iqr, self.eps)
        rec_robust = robust_standardize_with_ref(e_query, self.err_med, self.err_iqr, self.eps)
        if return_components:
            return {
                "surprise": surprise.astype(np.float32),
                "rarity": rarity.astype(np.float32),
                "rec_robust": rec_robust.astype(np.float32),
                "surprise_raw": raw["residual_z"].astype(np.float32),
                "surprise_soft_raw": raw["surprise_soft_raw"].astype(np.float32),
                "rarity_raw": raw["rarity_raw"].astype(np.float32),
                "tau": raw["tau"].astype(np.float32),
                "sigma": raw["sigma"].astype(np.float32),
            }
        return surprise.astype(np.float32)


def choose_fusion_alpha(y_val, sur_val, rare_val, alpha_values=None, num_thresholds=201):
    from rnec.metrics import best_f1_threshold

    y_val = np.asarray(y_val).astype(np.int64).reshape(-1)
    sur_val = np.asarray(sur_val, dtype=np.float64).reshape(-1)
    rare_val = np.asarray(rare_val, dtype=np.float64).reshape(-1)
    if alpha_values is None:
        alpha_values = np.linspace(0.0, 1.0, 21)
    best = None
    for alpha in alpha_values:
        alpha = float(alpha)
        s = alpha * sur_val + (1.0 - alpha) * rare_val
        p, r, f1, thr = best_f1_threshold(y_val, s, num_thresholds=num_thresholds)
        item = (float(f1), alpha, float(thr), float(p), float(r))
        if best is None or item[0] > best[0]:
            best = item
    return {
        "alpha": float(best[1]),
        "threshold": float(best[2]),
        "val_f1": float(best[0]),
        "val_precision": float(best[3]),
        "val_recall": float(best[4]),
    }
