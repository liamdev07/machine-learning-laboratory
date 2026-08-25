# Practical 07 — Feature Reduction with PCA

**Syllabus (Parul University):** *Perform feature reduction using PCA and visualise the transformed features.*  
**Kernel:** `ml-lab-kernel`  
**Data:** Wisconsin Breast Cancer (`data/breast_cancer.csv`) — 30 numeric assays, binary `diagnosis`

## Objectives

1. Mean-center **four** 2-D points; form \(\Sigma\); solve \(\det(\Sigma-\lambda I)=0\); recover unit \(v_1,v_2\); match NumPy / sklearn.
2. State **unsupervised** PCA, **eigendecomposition vs SVD**, and **reconstruction MSE**.
3. Implement `NumpyPCA` (`np.linalg.eigh`) whose eigenvalues and (sign-aligned) scores match `sklearn.decomposition.PCA`.
4. Fit a leakage-safe `Pipeline(StandardScaler, PCA)` on Breast Cancer; never scale on the test fold.
5. Plot **scree + 95%**, **2-D / 3-D scores**, **loadings heatmap + biplot**, and a **downstream classifier** (fit time and test accuracy before vs after PCA).

## Files

| Path | Role |
|------|------|
| `practical_07.ipynb` | Full lab |
| `data/breast_cancer.csv` | 30 assays + `diagnosis` |

## Swap your CSV (CONFIG cell)

```python
DATA_PATH = "data/breast_cancer.csv"
TARGET_COLUMN = "diagnosis"
NUMERIC_FEATURES = None    # or a list of numeric names
N_COMPONENTS = 2           # 2-D plots / biplot; scree still uses all PCs
DROP_COLUMNS = []
```

## Dataset loading (fallback)

1. Local `DATA_PATH`  
2. Colab Drive / upload  
3. `sklearn.datasets.load_breast_cancer`

## How to run

```bash
cd 07_PCA_Feature_Reduction
jupyter notebook practical_07.ipynb
```

Select **Python (ML Lab Kernel)**.

## Assessment hints

- Toy: \(\lambda_1=3\), \(\lambda_2=1/3\), \(\pi_1=0.9\), \(v_1=(1,1)/\sqrt{2}\).
- PCA does **not** use `diagnosis`; colour is display-only.
- Quote **test** accuracy at the \(q\) you chose; reconstruction MSE is not a class label.
- Unscaled PCA is dominated by large-range columns (area vs smoothness).
