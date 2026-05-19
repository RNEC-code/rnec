from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from rnec.calibration import LocalEvidenceCalibrator, choose_fusion_alpha
from rnec.ds_reader import Dataset_Loader
from rnec.metrics import as_window_labels, evaluate_with_val_threshold
from rnec.models import PerformerAutoEncoder

METHOD_VERSION = "rnec_release_v1"


@dataclass
class ExperimentConfig:
    dataset: str = "MSL"
    data_path: str = "./data/MSL"
    ts_num: int = 0
    seed: int = 42
    window_size: int = 128
    step_size: int = 1
    downsample_factor: int = 1
    downsample_method: str = "stride"
    val_ratio: float = 0.2
    split_seed: int = 42
    batch_size: int = 128
    num_workers: int = 0
    backbone: str = "performer"
    hidden: int = 128
    latent_dim: int = 48
    dropout: float = 0.1
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    grad_clip_norm: float = 1.0
    check_finite_grad: bool = False
    k_neighbors: int = 20
    tau_quantile: float = 0.95
    softplus_temp: float = 1.0
    max_ref: int = 20000
    threshold_steps: int = 201
    readout_window: int = 129
    device: str = "cuda"
    amp: bool = False
    outdir: str = "./runs/calae_performer"
    method_version: str = METHOD_VERSION


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str):
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def safe_to_csv(df: pd.DataFrame, path: str, index: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


def reconstruction_error(x, x_hat):
    return torch.mean((x_hat - x) ** 2, dim=(1, 2))


def clean_score(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    min_f = float(np.min(x[finite]))
    max_f = float(np.max(x[finite]))
    return np.nan_to_num(x, nan=min_f, posinf=max_f, neginf=min_f)


def robust_iqr(x):
    x = clean_score(x)
    q75, q25 = np.percentile(x, [75, 25])
    return ((x - np.median(x)) / max(float(q75 - q25), 1e-8)).astype(np.float32)


def robust_mad(x):
    x = clean_score(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return ((x - med) / max(float(mad), 1e-8)).astype(np.float32)


def roll_max(x, window):
    return pd.Series(clean_score(x)).rolling(int(window), center=True, min_periods=1).max().to_numpy().astype(np.float32)


def amp_enabled(cfg: ExperimentConfig, device) -> bool:
    return bool(cfg.amp and device.type == "cuda")


def make_grad_scaler(cfg: ExperimentConfig, device):
    return torch.amp.GradScaler("cuda", enabled=amp_enabled(cfg, device))


def has_nonfinite_grad(model) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def build_model(in_channels: int, cfg: ExperimentConfig):
    backbone = str(cfg.backbone).lower()
    if backbone != "performer":
        raise ValueError("The clean RNEC release supports only the Performer AE backbone.")
    return PerformerAutoEncoder(
        in_channels=in_channels,
        window_size=cfg.window_size,
        hidden=cfg.hidden,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_ff=(cfg.dim_ff if int(cfg.dim_ff) > 0 else None),
    )


def train_plain_ae(model, train_loader, cfg: ExperimentConfig, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_grad_scaler(cfg, device)
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        t0 = time.time()
        losses = []
        skipped = 0
        model.train()
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled(cfg, device)):
                x_hat, _ = model(x)
                loss = torch.mean((x_hat - x) ** 2)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(opt)
                if cfg.check_finite_grad and has_nonfinite_grad(model):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        row = {"epoch": ep, "loss": mean_loss, "skipped_batches": skipped, "sec": time.time() - t0}
        history.append(row)
        print(f"[AE] epoch={ep:03d}/{cfg.epochs:03d} loss={row['loss']:.6f} skipped={skipped} sec={row['sec']:.1f}")
        if not np.isfinite(mean_loss):
            raise FloatingPointError("AE training produced only non-finite losses.")
    return history


@torch.no_grad()
def collect_stats(model, loader, device, has_labels: bool):
    model.eval()
    zs, es, ys = [], [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        x_hat, z = model(x)
        zs.append(z.detach().cpu().numpy().astype(np.float32))
        es.append(reconstruction_error(x, x_hat).detach().cpu().numpy().astype(np.float32))
        if has_labels:
            ys.append(as_window_labels(batch[1].detach().cpu().numpy()))
    out = {"z": np.concatenate(zs, axis=0), "e": np.concatenate(es, axis=0)}
    if has_labels:
        out["y"] = np.concatenate(ys, axis=0).astype(np.int64)
    return out


def fit_calibrator_and_score(train_stats, val_stats, test_stats, cfg: ExperimentConfig):
    cal = LocalEvidenceCalibrator(
        k=cfg.k_neighbors,
        tau_quantile=cfg.tau_quantile,
        max_ref=cfg.max_ref,
        seed=cfg.seed,
        softplus_temp=cfg.softplus_temp,
    )
    cal.fit(train_stats["z"], train_stats["e"])
    val_comp = cal.score(val_stats["z"], val_stats["e"], return_components=True)
    test_comp = cal.score(test_stats["z"], test_stats["e"], return_components=True)
    return val_comp, test_comp


def save_window_scores(path, starts, y, e, score, comp=None):
    df = pd.DataFrame(
        {
            "window_start": np.asarray(starts, dtype=np.int64),
            "label": np.asarray(y, dtype=np.int64),
            "rec_error": np.asarray(e, dtype=float),
            "score": np.asarray(score, dtype=float),
        }
    )
    if comp is not None:
        for key in ["rec_robust", "surprise", "rarity", "surprise_raw", "surprise_soft_raw", "rarity_raw", "tau", "sigma"]:
            if key in comp:
                df[key] = np.asarray(comp[key], dtype=float)
    safe_to_csv(df, path, index=False)


def add_metric_metadata(metrics, cfg, series_name, method, score_type, readout, alpha=None, alpha_val_f1=None):
    metrics.update(
        {
            "method": method,
            "dataset": cfg.dataset,
            "series": series_name,
            "ts_num": cfg.ts_num,
            "seed": cfg.seed,
            "backbone": cfg.backbone,
            "score_type": score_type,
            "readout": readout,
            "readout_window": int(cfg.readout_window) if readout != "none" else 0,
            "method_version": cfg.method_version,
        }
    )
    if alpha is not None:
        metrics["alpha_surprise"] = alpha
    if alpha_val_f1 is not None:
        metrics["alpha_val_f1"] = alpha_val_f1
    return metrics


def run_one_series(cfg: ExperimentConfig) -> pd.DataFrame:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    series_dir = os.path.join(cfg.outdir, f"seed_{cfg.seed}", f"ts_{cfg.ts_num}")
    os.makedirs(series_dir, exist_ok=True)
    with open(os.path.join(series_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    loader_obj = Dataset_Loader(
        dataset=cfg.dataset,
        data_path=cfg.data_path,
        window_size=cfg.window_size,
        ts_num=cfg.ts_num,
        step_size=cfg.step_size,
        val_ratio=cfg.val_ratio,
        downsample_factor=cfg.downsample_factor,
        downsample_method=cfg.downsample_method,
        split_seed=cfg.split_seed,
    )
    series_name = getattr(loader_obj, "series_name", str(cfg.ts_num))
    in_channels = int(loader_obj.dataset["train_data"].shape[0])
    train_loader = loader_obj.train_loader_generation(cfg.batch_size, shuffle=True, seed=cfg.seed, num_workers=cfg.num_workers)
    val_loader = loader_obj.val_loader_generation(cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = loader_obj.test_loader_generation(cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(in_channels, cfg)
    print(f"\n===== Train AE: dataset={cfg.dataset} series={series_name} seed={cfg.seed} =====")
    hist_ae = train_plain_ae(model, train_loader, cfg, device)
    safe_to_csv(pd.DataFrame(hist_ae), os.path.join(series_dir, "history_ae.csv"))
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "series_name": series_name}, os.path.join(series_dir, "ae.pt"))

    train_stats = collect_stats(model, train_loader, device, has_labels=False)
    val_stats = collect_stats(model, val_loader, device, has_labels=True)
    test_stats = collect_stats(model, test_loader, device, has_labels=True)

    rows = []
    ae_metrics = add_metric_metadata(
        evaluate_with_val_threshold(val_stats["y"], val_stats["e"], test_stats["y"], test_stats["e"], cfg.threshold_steps),
        cfg,
        series_name,
        "AE",
        "raw_rec",
        "none",
    )
    rows.append(ae_metrics)
    save_window_scores(os.path.join(series_dir, "scores_AE_test.csv"), loader_obj.test_window_starts, test_stats["y"], test_stats["e"], test_stats["e"])

    ae_max_val = robust_iqr(roll_max(val_stats["e"], cfg.readout_window))
    ae_max_test = robust_iqr(roll_max(test_stats["e"], cfg.readout_window))
    ae_max_metrics = add_metric_metadata(
        evaluate_with_val_threshold(val_stats["y"], ae_max_val, test_stats["y"], ae_max_test, cfg.threshold_steps),
        cfg,
        series_name,
        "AE+max129",
        "raw_rec",
        "max129",
    )
    rows.append(ae_max_metrics)
    save_window_scores(
        os.path.join(series_dir, "scores_AE_max129_test.csv"),
        loader_obj.test_window_starts,
        test_stats["y"],
        test_stats["e"],
        ae_max_test,
    )

    print(f"\n===== Fit calibration on frozen AE: dataset={cfg.dataset} series={series_name} seed={cfg.seed} =====")
    val_comp, test_comp = fit_calibrator_and_score(train_stats, val_stats, test_stats, cfg)
    alpha_info = choose_fusion_alpha(val_stats["y"], val_comp["surprise"], val_comp["rarity"], num_thresholds=cfg.threshold_steps)
    alpha = alpha_info["alpha"]
    s_val = alpha * val_comp["surprise"] + (1.0 - alpha) * val_comp["rarity"]
    s_test = alpha * test_comp["surprise"] + (1.0 - alpha) * test_comp["rarity"]
    cal_metrics = add_metric_metadata(
        evaluate_with_val_threshold(val_stats["y"], s_val, test_stats["y"], s_test, cfg.threshold_steps),
        cfg,
        series_name,
        "AE+Calibration",
        "surprise_plus_rarity",
        "none",
        alpha=alpha,
        alpha_val_f1=alpha_info["val_f1"],
    )
    rows.append(cal_metrics)
    save_window_scores(os.path.join(series_dir, "scores_AE_Calibration_test.csv"), loader_obj.test_window_starts, test_stats["y"], test_stats["e"], s_test, test_comp)

    cal_base_val = alpha * robust_mad(val_comp["surprise"]) + (1.0 - alpha) * robust_mad(val_comp["rarity"])
    cal_base_test = alpha * robust_mad(test_comp["surprise"]) + (1.0 - alpha) * robust_mad(test_comp["rarity"])
    cal_max_val = robust_iqr(roll_max(cal_base_val, cfg.readout_window))
    cal_max_test = robust_iqr(roll_max(cal_base_test, cfg.readout_window))
    cal_max_metrics = add_metric_metadata(
        evaluate_with_val_threshold(val_stats["y"], cal_max_val, test_stats["y"], cal_max_test, cfg.threshold_steps),
        cfg,
        series_name,
        "AE+Calibration+max129",
        "surprise_plus_rarity",
        "max129",
        alpha=alpha,
        alpha_val_f1=alpha_info["val_f1"],
    )
    rows.append(cal_max_metrics)
    save_window_scores(
        os.path.join(series_dir, "scores_AE_Calibration_max129_test.csv"),
        loader_obj.test_window_starts,
        test_stats["y"],
        test_stats["e"],
        cal_max_test,
        test_comp,
    )

    result = pd.DataFrame(rows)
    safe_to_csv(result, os.path.join(series_dir, "metrics.csv"))
    return result
