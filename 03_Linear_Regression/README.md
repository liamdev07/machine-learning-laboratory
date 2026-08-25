# Practical 03 — Linear Regression and Residual Error

**University mapping:** *Implement Linear Regression and compute residual error.*  
**Duration:** 1 lab session (+ derivation take-home)  
**Stack:** NumPy (hand OLS + GD), Pandas, Matplotlib, Seaborn, SciPy (Q–Q), scikit-learn  
**Kernel:** `ml-lab-kernel`

## Objectives

1. Compute **slope** $m$, **intercept** $c$, $\hat{y}_i$, residuals $e_i=y_i-\hat{y}_i$, $SS_{\mathrm{res}}$, $SS_{\mathrm{tot}}$, and $R^2$ on a **5-row** table; match NumPy.
2. State the **OLS normal equations** $\hat\beta=(X^{\top}X)^{-1}X^{\top}y$ (solve; do not invert in code) and a **batch GD** update.
3. State four modelling assumptions: **linearity**, **homoscedasticity**, **normality of residuals**, **independence**.
4. Fit `sklearn.linear_model.LinearRegression` with `train_test_split` and **train-only** `StandardScaler`.
5. Report **MAE, MSE, RMSE, $R^2$, adjusted $R^2$** and a **4-panel** residual figure.

## Files

| Path | Role |
|------|------|
| `practical_03.ipynb` | Full experiment (config, hand OLS, GD, sklearn, diagnostics, viva) |
| `data/medical_insurance.csv` | Medical Cost Personal data (`charges` in USD) |

**Citation:** Lantz, *Machine Learning with R*, companion `insurance.csv`.

## Swap your CSV (CONFIG cell)

Edit **only** the **DATASET CONFIG** block at the top of the notebook:

```python
DATA_PATH = "data/medical_insurance.csv"
TARGET_COLUMN = "charges"
FEATURE_COLUMNS = ["age", "bmi", "children"]  # numeric only
DROP_COLUMNS = []
```

Headers must be on the first row. `FEATURE_COLUMNS` must be numeric for `StandardScaler`.

## Dataset loading (ordered fallback)

1. Local `DATA_PATH`
2. Colab Drive / upload
3. GitHub raw `insurance.csv` (Stedy / Lantz mirror)
4. `sklearn.datasets.fetch_california_housing` (different domain — record origin)

## How to run

```bash
cd 03_Linear_Regression
jupyter notebook practical_03.ipynb
```

## Assessment hints

- OLS residuals are orthogonal to the columns of $X$ (including the intercept).
- Scaling changes the GD bowl; fit the scaler on **train** only.
- Quote **test** RMSE in the lab record, not only training $R^2$.
- A fan or curve in residuals vs fitted violates homoscedasticity / linearity even if $R^2$ looks fine.
