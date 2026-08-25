# Practical 01 — Python ML Basics

**Course:** M.Tech Machine Learning Laboratory  
**Duration:** 1 lab session (+ take-home review)  
**Stack:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn  
**Kernel:** `ml-lab-kernel` (Python ML Lab Kernel)

## Objectives

1. Use **vectorized** NumPy (no Python loops for numeric kernels), including broadcasting and a tiny OLS identity.
2. Load, inspect, filter, and aggregate tabular data with **Pandas**.
3. Produce **clear** EDA figures (distributions, scatter, correlation heatmap).
4. Build a **leakage-safe** sklearn baseline: `train_test_split` → `SimpleImputer` + `StandardScaler` (fit on train only) → linear model → hold-out metrics.

## Files

| Path | Role |
|------|------|
| `practical_01.ipynb` | Full experiment (config, theory, code, viva, conclusion) |
| `data/california_housing_lab_benchmark.csv` | Local 400-row housing-style benchmark (seed-generated, version-controlled) |

## Swap your own CSV (3 lines)

Open the notebook and edit **only** the **DATASET CONFIG BLOCK** (near the top, after the Colab header):

```python
DATA_PATH = "data/california_housing_lab_benchmark.csv"  # or your file
TARGET_COLUMN = "median_house_value"                     # must match a header
DROP_COLUMNS = []                                        # e.g. ["id"]
```

The first row of the CSV must be column names. If `TARGET_COLUMN` is missing, the notebook **stops** and prints the real headers.

## Dataset loading (ordered fallback)

1. **Local:** `DATA_PATH` (this folder’s `data/` file by default).
2. **Colab Drive:** if `COLAB_DRIVE_CSV` is set, mount Drive and read that path.
3. **Colab upload:** file picker (`files.upload()`) if the local path is missing.
4. **URL:** public California Housing CSV (Hands-On ML companion data, Aurélien Géron).
5. **Last resort:** `sklearn.datasets.fetch_california_housing` as a DataFrame.

Always record `load_origin` in your lab file. The **400-row lab CSV** and the **~20k** Géron/sklearn table are **not** the same experiment — metrics will differ.

## How to run

From this directory, with the root `requirements.txt` installed and kernel `ml-lab-kernel`:

```bash
jupyter notebook practical_01.ipynb
```

**Google Colab:** open the notebook → run the **Colab header** (`%pip install` + upload/Drive helpers) → run **DATASET CONFIG**. You do not have to upload `data/` if you accept the URL/sklearn fallback.

### Troubleshooting

| Symptom | What to do |
|---------|------------|
| File not found | Fix `DATA_PATH` (relative to the notebook). On Colab, upload or set `COLAB_DRIVE_CSV`. |
| `TARGET_COLUMN` error | Copy a name from the printed `columns` list. |
| SSL / URL failed | Use the local CSV, an upload, or the sklearn last resort. |
| Metrics ≠ classmate | Compare `load_origin`, `SEED`, and library versions. |

## Assessment hints

- Explain **why** scaling and median imputation are fit on the training split only.
- Contrast vectorized inner products with nested Python loops (see the timing table).
- Interpret a correlation heatmap without claiming causation.
