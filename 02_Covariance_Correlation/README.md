# Practical 02 — Covariance and Correlation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/02_Covariance_Correlation/practical_02.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/NumPy%20%7C%20Pandas%20%7C%20Matplotlib-013243?style=flat-square)](https://pandas.pydata.org/)

> How two features move together — the statistic behind feature selection, PCA and multicollinearity.

## Aim & Objectives

**Aim:** To compute the covariance and the Pearson correlation between the features of a dataset.

**Objectives**

1. Load the Iris dataset and separate its numeric feature columns.
2. Compute the covariance matrix and the correlation matrix using Pandas.
3. Display both matrices as annotated Matplotlib heatmaps.
4. Explain why correlation is preferred when features use different units.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/iris.csv` |
| Size | 150 rows × 5 columns |
| Features | `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm` |
| Label column | `species` (dropped before computing covariance) |
| Extra data | `data/auto_mpg.csv` for optional practice |

```python
CSV = "data/iris.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/02_Covariance_Correlation/data/iris.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → drop label → `.cov()` → `.corr()` → two heatmaps |
| 3 | Line-by-Line Code Explanation | Every import, function and variable in plain language |
| 4 | Output & Graph Interpretation | What the two heatmaps reveal about the four features |

## Expected outputs & results

| Pair | Covariance | Correlation |
|------|-----------:|------------:|
| petal length ↔ petal width | 1.296 | **0.963** |
| sepal length ↔ petal length | 1.274 | 0.872 |
| sepal width ↔ petal length | −0.330 | −0.428 |

**Figure — two annotated heatmaps**

- *Left:* covariance matrix (`viridis`), values still carry cm² units, so the scale looks uneven.
- *Right:* correlation matrix (`coolwarm`), every value between −1 and +1 with 1.0 on the diagonal.

## Viva Q&A highlights

**Q1. What is the difference between covariance and correlation?**
Covariance keeps the units of the two variables, so its size is hard to judge. Correlation divides covariance by the two standard deviations, giving a unit-free value between −1 and +1.

**Q2. Why is the diagonal of the correlation matrix always 1?**
A feature is perfectly correlated with itself; on the diagonal the correlation formula reduces to variance divided by variance.

**Q3. What does a negative correlation mean here?**
As sepal width increases, petal length tends to decrease — the two move in opposite directions.

**Q4. Petal length and petal width correlate at 0.96. Is that a problem?**
For interpretation it means both features carry nearly the same information (multicollinearity); one of them, or a PCA component, is often enough.

**Q5. Why must the `species` column be dropped first?**
Covariance is defined for numeric variables. `species` is text, so Pandas cannot compute a meaningful product of deviations for it.

## Run it locally

```bash
cd 02_Covariance_Correlation
jupyter notebook practical_02.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`.
