# Practical 02 — Covariance and Correlation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/02_Covariance_Correlation/practical_02.ipynb)

**Aim:** To compute the covariance and the Pearson correlation between the features of a dataset.

**Dataset:** `data/iris.csv` — 150 flowers, 4 numeric measurements and the `species` column.

## Objectives

1. Load the Iris dataset and separate its numeric feature columns.
2. Compute the covariance matrix and the correlation matrix using Pandas.
3. Display both matrices as Matplotlib heatmaps and compare them.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the two heatmaps show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 02_Covariance_Correlation
jupyter notebook practical_02.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`.

## Expected result

Petal length and petal width are almost perfectly related (r ≈ 0.96), while sepal width is slightly negatively related to the petal features.

## Viva points

- Covariance keeps the units of the data; correlation is unit-free and always lies between −1 and +1.
- Why the diagonal of a correlation matrix is always 1.
- A strong correlation shows association, not causation.

`data/auto_mpg.csv` is also kept in this folder if you wish to repeat the same steps on a second dataset.
