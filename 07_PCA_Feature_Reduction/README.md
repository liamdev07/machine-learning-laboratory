# Practical 07 — Feature Reduction using PCA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/07_PCA_Feature_Reduction/practical_07.ipynb)

**Aim:** To perform feature reduction using Principal Component Analysis and visualise the transformed features.

**Dataset:** `data/breast_cancer.csv` — 569 tumours, 30 numeric measurements, `diagnosis` = malignant / benign.

## Objectives

1. Standardise the 30 features of the Breast Cancer dataset.
2. Apply PCA and study the explained variance of each principal component.
3. Project the data onto 2 components, plot it, and compare accuracy before and after PCA.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the scree plot and 2-D projection show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 07_PCA_Feature_Reduction
jupyter notebook practical_07.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Expected result

The first two components keep about 63% of the variance and 10 components reach the 95% level; accuracy falls only from about 0.97 (30 features) to 0.94 (2 components).

## Viva points

- PCA is **unsupervised** — the `diagnosis` label is used only to colour the plot.
- Standardisation is required first, otherwise `mean area` dominates every component.
- Explained variance ratio and the meaning of the cumulative (scree) curve.
- Reducing 30 columns to 2 loses some accuracy but makes the data easy to visualise.
