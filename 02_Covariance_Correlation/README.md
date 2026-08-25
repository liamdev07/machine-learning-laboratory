# Practical 02 — Covariance and Correlation

**Course:** M.Tech Machine Learning Laboratory  
**Duration:** 1 lab session (+ take-home derivation)  
**Stack:** NumPy, Pandas, Matplotlib, Seaborn  
**Kernel:** `ml-lab-kernel`

## Objectives

1. Compute **sample covariance** with Bessel’s correction \(n-1\) by hand (means → deviations → products → sums of squares → \(s_{xy}\) → Pearson \(r\)).
2. Match that hand table against **NumPy** (`ddof=1`) and **Pandas**.
3. Form the **variance–covariance matrix** \(\mathbf{S}=\tilde{X}^{\top}\tilde{X}/(n-1)\) and the **correlation matrix** \(\mathbf{R}\) (symmetry, diagonal variances / ones, scale-invariance of \(r\)).
4. Point the notebook at a CSV with **DATASET CONFIG**: path, numeric features, optional target, drop list.
5. Draw publication-grade figures: **+r / −r / 0** scatter with OLS trendlines, annotated **`coolwarm` heatmap** (`vmin=-1`, `vmax=1`), **pairplot**, and a real negative \(r\) (weight vs mpg).

## Files

| Path | Role |
|------|------|
| `practical_02.ipynb` | Full experiment (config, hand calculation, matrices, plots, viva) |
| `data/iris.csv` | Fisher / sklearn Iris (150 rows, 4 numeric traits + species) |
| `data/auto_mpg.csv` | Auto MPG (fuel economy vs engine/vehicle attributes) |

## Swap your CSV (CONFIG cell)

Edit **only** the **DATASET CONFIG** cell near the top:

```python
DATA_PATH = "data/iris.csv"          # or your file
TARGET_COLUMN = "species"            # colouring; excluded from covariance
DROP_COLUMNS = []                    # e.g. ["id"]
NUMERIC_FEATURES = None              # None = all numeric except target
# or: NUMERIC_FEATURES = ["petal_length_cm", "petal_width_cm"]
MPG_PATH = "data/auto_mpg.csv"
```

The first row of the CSV must be headers. Covariance uses **numeric** columns only.

## Dataset loading (ordered fallback)

1. **Local** `DATA_PATH` / `MPG_PATH`
2. **Colab** Drive path or file upload
3. **URL** (seaborn-data Iris; Plotly Auto MPG)
4. **Last resort (Iris):** `sklearn.datasets.load_iris`

Record `load_origin` in the lab file. Iris (150 rows) and Auto MPG (~398 rows) are different experiments.

## How to run

Select kernel **Python (ML Lab Kernel)** (`ml-lab-kernel`). From this directory:

```bash
jupyter notebook practical_02.ipynb
```

**Google Colab:** run the header cell (`%pip`, optional upload/Drive), then CONFIG, then Run all.

## Assessment hints

- Why Pearson \(r\) is invariant to positive affine rescaling, while covariance is not.
- Population \(1/N\) vs sample \(1/(n-1)\).
- \(r \approx 0\) is **not** independence (nonlinear dependence, small \(n\)).
- Heatmap colour is only comparable if the scale is locked to \([-1,1]\).
