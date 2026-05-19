from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rnec.train_eval import ExperimentConfig, run_one_series


REQUIRED_METHODS = {"AE", "AE+max129", "AE+Calibration", "AE+Calibration+max129"}
DATASET_SERIES_COUNTS = {
    "MSL": 28,
    "SMAP": 54,
    "psm": 1,
    "swat": 1,
    "wadi": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run RNEC on one of the five benchmark datasets.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--ts-nums", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--readout-window", type=int, default=None)
    return parser.parse_args()


def parse_int_list(text):
    if isinstance(text, (list, tuple)):
        if len(text) == 1 and str(text[0]).lower() == "all":
            return "all"
        return [int(x) for x in text]
    if str(text).strip().lower() == "all":
        return "all"
    return [int(x.strip()) for x in str(text).replace(",", " ").split() if x.strip()]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(args):
    cfg = load_config(args.config)
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key.replace("-", "_")] = value
    return cfg


def resolve_ts_nums(dataset, value):
    parsed = parse_int_list(value)
    if parsed != "all":
        return parsed
    key = str(dataset)
    key_norm = "MSL" if key.lower() == "msl" else "SMAP" if key.lower() == "smap" else key.lower()
    if key_norm not in DATASET_SERIES_COUNTS:
        raise ValueError(f"Cannot resolve ts_nums=all for dataset={dataset}.")
    return list(range(DATASET_SERIES_COUNTS[key_norm]))


def write_method_summary(summary, outdir):
    metric_cols = [
        "precision",
        "recall",
        "f1",
        "auroc",
        "aupr",
        "val_f1",
        "precision_oracle",
        "recall_oracle",
        "f1_oracle",
    ]
    rows = []
    for method, group in summary.groupby("method"):
        row = {"method": method, "n": int(len(group))}
        for col in metric_cols:
            if col in group.columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "summary_by_method.csv"), index=False)


def readable_method_table(summary, method):
    df = summary[summary["method"] == method].copy()
    if df.empty:
        return df
    preferred = [
        "dataset",
        "ts_num",
        "series",
        "seed",
        "method",
        "f1",
        "f1_oracle",
        "val_f1",
        "auroc",
        "aupr",
        "precision",
        "recall",
        "precision_oracle",
        "recall_oracle",
        "threshold_val",
        "threshold_oracle",
        "score_type",
        "alpha_surprise",
        "alpha_val_f1",
        "method_version",
    ]
    cols = [c for c in preferred if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols].sort_values(["dataset", "ts_num", "seed"], kind="stable")


def write_readable_method_tables(summary, outdir):
    readable_method_table(summary, "AE").to_csv(os.path.join(outdir, "summary_AE.csv"), index=False)
    readable_method_table(summary, "AE+max129").to_csv(os.path.join(outdir, "summary_AE_max129.csv"), index=False)
    readable_method_table(summary, "AE+Calibration").to_csv(os.path.join(outdir, "summary_AE_Calibration.csv"), index=False)
    readable_method_table(summary, "AE+Calibration+max129").to_csv(
        os.path.join(outdir, "summary_AE_Calibration_max129.csv"), index=False
    )


def read_existing_series_rows(series_dir):
    metrics_path = os.path.join(series_dir, "metrics.csv")
    if not os.path.isfile(metrics_path):
        return None
    try:
        metrics = pd.read_csv(metrics_path)
    except Exception:
        return None
    if "method" not in metrics.columns:
        return None
    if REQUIRED_METHODS.issubset(set(metrics["method"].astype(str))):
        return metrics
    return None


def run_config(cfg_dict):
    cfg_dict = dict(cfg_dict)
    cfg_dict["backbone"] = "performer"
    outdir = cfg_dict.get("outdir", "./runs/rnec")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    ts_nums = resolve_ts_nums(cfg_dict.get("dataset", "MSL"), cfg_dict.get("ts_nums", "all"))
    seeds = parse_int_list(cfg_dict.get("seeds", "42"))
    if seeds == "all":
        raise ValueError("seeds does not support 'all'.")

    for seed in seeds:
        for ts_num in ts_nums:
            series_dir = os.path.join(outdir, f"seed_{seed}", f"ts_{ts_num}")
            existing = read_existing_series_rows(series_dir)
            if existing is not None:
                print(f"[resume] skip completed series: ts_num={ts_num} seed={seed}")
                rows.extend(existing.to_dict("records"))
            else:
                cfg = ExperimentConfig(
                    dataset=cfg_dict.get("dataset", "MSL"),
                    data_path=cfg_dict.get("data_path", "./data/MSL"),
                    ts_num=ts_num,
                    seed=seed,
                    outdir=outdir,
                    window_size=int(cfg_dict.get("window_size", 128)),
                    step_size=int(cfg_dict.get("step_size", 1)),
                    downsample_factor=int(cfg_dict.get("downsample_factor", 1)),
                    downsample_method=cfg_dict.get("downsample_method", "stride"),
                    val_ratio=float(cfg_dict.get("val_ratio", 0.2)),
                    split_seed=int(cfg_dict.get("split_seed", 42)),
                    batch_size=int(cfg_dict.get("batch_size", 128)),
                    backbone="performer",
                    epochs=int(cfg_dict.get("epochs", 10)),
                    latent_dim=int(cfg_dict.get("latent_dim", 48)),
                    hidden=int(cfg_dict.get("hidden", 128)),
                    dropout=float(cfg_dict.get("dropout", 0.1)),
                    nhead=int(cfg_dict.get("nhead", 4)),
                    num_layers=int(cfg_dict.get("num_layers", 2)),
                    lr=float(cfg_dict.get("lr", 1e-3)),
                    readout_window=int(cfg_dict.get("readout_window", 129)),
                    k_neighbors=int(cfg_dict.get("k_neighbors", 20)),
                    tau_quantile=float(cfg_dict.get("tau_quantile", 0.95)),
                    softplus_temp=float(cfg_dict.get("softplus_temp", 1.0)),
                    check_finite_grad=bool(cfg_dict.get("check_finite_grad", False)),
                    device=cfg_dict.get("device", "cuda"),
                    amp=bool(cfg_dict.get("amp", False)),
                )
                rows.extend(run_one_series(cfg).to_dict("records"))

            summary = pd.DataFrame(rows)
            summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
            write_method_summary(summary, outdir)
            write_readable_method_tables(summary, outdir)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    cfg = merge_config(args)
    run_config(cfg)


if __name__ == "__main__":
    main()
