# Practical 06 — k-Nearest Neighbours (kNN)

**Syllabus (Parul University):** *Implement k-Nearest Neighbours (kNN) and evaluate accuracy.*  
**Kernel:** `ml-lab-kernel`  
**Data:** UCI Wine (`data/wine.csv`) — 13 chemical features, 3 cultivars

## Objectives

1. Compute **Euclidean** distances from a query to **six** 2-D training points; vote for $k=1,3,5$ with **uniform** and **inverse-distance** weights; match sklearn.
2. State **lazy / non-parametric** learning, the **bias–variance** role of $k$, and **train-only** `StandardScaler`.
3. Implement a vectorized NumPy `KNNClassifier` whose labels match `KNeighborsClassifier` (brute, Euclidean).
4. Wrap sklearn kNN in a **Pipeline** with scaler fit on train; sweep **odd** $k\in\{1,3,\ldots,25\}$ on a **validation** set; pick $k^\star$.
5. Plot **train vs val error vs $k$**, **three 2-D decision regions** ($k=1$, $k^\star$, large $k$), and a **test confusion matrix**.

## Files

| Path | Role |
|------|------|
| `practical_06.ipynb` | Full lab |
| `data/wine.csv` | UCI Wine + `cultivar` |

## Swap your CSV (CONFIG cell)

```python
DATA_PATH = "data/wine.csv"
TARGET_COLUMN = "cultivar"
FEATURE_COLUMNS = None       # or a list of numeric names
K_NEIGHBORS = 5              # default k; grid search still runs 1..25 odd
DROP_COLUMNS = []
```

## Dataset loading (fallback)

1. Local `DATA_PATH`  
2. Colab Drive / upload  
3. `sklearn.datasets.load_wine`

## How to run

```bash
cd 06_KNN_Classifier
jupyter notebook practical_06.ipynb
```

Select **Python (ML Lab Kernel)**.

## Assessment hints

- $k=3$ on the toy: uniform **B**, distance-weighted **A** — explain why.
- Never choose $k$ on the test set; quote **test** accuracy at $k^\star$.
- Unscaled Euclidean is dominated by large-range columns (e.g. proline).
- $k=1$ train error $\approx 0$ is not a success; look at validation.
