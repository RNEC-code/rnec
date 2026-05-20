# RNEC Clean Release

This is a minimal release package for reproducing the five-dataset RNEC
experiments with the Performer autoencoder backbone.

RNEC is a post-training reference-guided calibration layer over a trained AE.


## Directory Layout

```text
rnec_release_clean/
  rnec/
    models.py
    calibration.py
    ds_reader.py
    metrics.py
    train_eval.py
  configs/
    msl.yaml
    smap.yaml
    psm.yaml
    swat.yaml
    wadi.yaml
  run_rnec.py
  run_all_five_datasets.ps1
  summarize_five_datasets.py
  requirements.txt
```

## Data Layout

The default configs expect data next to this release folder:

```text
../data/MSL
../data/PSM
../data/SWaT
../data/WADI
```

In this local workspace, SMAP files are stored under `../data/MSL`, so
`configs/smap.yaml` intentionally uses `data_path: ../data/MSL`.

## Run One Dataset

```powershell
python run_rnec.py --config configs/msl.yaml
python run_rnec.py --config configs/smap.yaml
python run_rnec.py --config configs/psm.yaml
python run_rnec.py --config configs/swat.yaml
python run_rnec.py --config configs/wadi.yaml
```

To run all five sequentially:

```powershell
./run_all_five_datasets.ps1
```

## Outputs

Each series writes:

```text
runs/<dataset>_full/seed_42/ts_<id>/
  ae.pt
  config.json
  history_ae.csv
  metrics.csv
  scores_AE_test.csv
  scores_AE_max129_test.csv
  scores_AE_Calibration_test.csv
  scores_AE_Calibration_max129_test.csv
```

Dataset-level summaries:

```text
runs/<dataset>_full/summary.csv
runs/<dataset>_full/summary_by_method.csv
runs/<dataset>_full/summary_AE.csv
runs/<dataset>_full/summary_AE_max129.csv
runs/<dataset>_full/summary_AE_Calibration.csv
runs/<dataset>_full/summary_AE_Calibration_max129.csv
```

After all five datasets finish:

```powershell
python summarize_five_datasets.py
```

This writes:

```text
summary_tables/five_dataset_method_summary.csv
summary_tables/five_dataset_method_average.csv
```



