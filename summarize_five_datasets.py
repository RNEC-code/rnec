from __future__ import annotations

from pathlib import Path

import pandas as pd


RUNS_ROOT = Path("runs")
OUT_DIR = Path("summary_tables")
DATASETS = ["msl", "smap", "psm", "swat", "wadi"]
METRICS = ["f1_mean", "auroc_mean", "aupr_mean", "f1_oracle_mean"]


def load_long_table() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        path = RUNS_ROOT / f"{ds}_full" / "summary_by_method.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df.insert(0, "dataset", ds.upper() if ds in {"msl", "smap"} else ds)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    keep = ["dataset", "method", "n"] + METRICS
    return out[keep].sort_values(["dataset", "method"], kind="stable")


def write_method_average(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method", sort=False):
        row = {"method": method, "n_datasets": int(len(group))}
        for metric in METRICS:
            row[f"{metric}_avg"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "five_dataset_method_average.csv", index=False)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long = load_long_table()
    long.to_csv(OUT_DIR / "five_dataset_method_summary.csv", index=False)
    avg = write_method_average(long)
    print(f"wrote {OUT_DIR / 'five_dataset_method_summary.csv'}")
    print(f"wrote {OUT_DIR / 'five_dataset_method_average.csv'}")
    print(avg.to_string(index=False))


if __name__ == "__main__":
    main()
