# Practical 07 — Feature Reduction using PCA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/07_PCA_Feature_Reduction/practical_07.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-PCA-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> Thirty correlated tumour measurements compressed into two components you can actually plot — with almost no loss in accuracy.

## Aim & Objectives

**Aim:** To perform feature reduction using Principal Component Analysis and visualise the transformed features.

**Objectives**

1. Standardise the 30 numeric features of the Breast Cancer dataset.
2. Apply PCA and study how much variance each principal component explains.
3. Find how many components are needed to keep 95% of the information.
4. Project onto 2 components, plot the result, and compare classifier accuracy before and after PCA.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/breast_cancer.csv` |
| Size | 569 rows × 31 columns |
| Features | 30 numeric measurements (`mean radius` … `worst fractal dimension`) |
| Target | `diagnosis` — `malignant` / `benign` (used only to colour the plot) |
| Preprocessing | `StandardScaler` before PCA — mandatory |

```python
CSV = "data/breast_cancer.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/07_PCA_Feature_Reduction/data/breast_cancer.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → scale → PCA(all) → PCA(2) → accuracy comparison → two plots |
| 3 | Line-by-Line Code Explanation | Explained variance, cumulative sum and every Scikit-Learn call |
| 4 | Output & Graph Interpretation | What the scree curve and the 2-D projection show |

## Expected outputs & results

| Quantity | Value |
|----------|------:|
| Variance explained by PC1 | 0.4427 |
| Variance explained by PC2 | 0.1897 |
| Variance kept by 2 components | 0.6324 |
| Components needed for 95% | 10 |
| Accuracy with all 30 features | 0.9708 |
| Accuracy with 2 PCA components | 0.9415 |

**Figure — two panels**

- *Left:* scree plot — bars for individual variance, an orange cumulative curve, and a red dashed 95% line.
- *Right:* the data projected on PC1 vs PC2, where malignant and benign form two clearly separated clouds.

## Viva Q&A highlights

**Q1. Is PCA supervised or unsupervised?**
Unsupervised. The `diagnosis` label never enters the computation; it is used only to colour the scatter plot.

**Q2. Why standardise before PCA?**
PCA maximises variance. Without scaling, `mean area` (in hundreds) would carry more raw variance than `mean smoothness` (about 0.1) and would hijack the first component.

**Q3. What is the explained variance ratio?**
The share of total variance captured by each component, λⱼ / Σλ. The components are ordered so the first captures the most.

**Q4. What does a principal component actually represent?**
A new axis built as a weighted combination of the original features, chosen to be orthogonal to the previous ones and to capture the largest remaining variance.

**Q5. Accuracy dropped from 0.97 to 0.94 — was PCA worth it?**
Often yes: 30 features became 2, so the model is far smaller, faster and easy to visualise. The trade-off between compression and accuracy is the decision you must justify.

**Q6. How do you choose the number of components?**
Either the elbow of the scree plot, or the smallest count reaching a variance threshold such as 95% (10 components here), or whichever value maximises downstream accuracy.

## Run it locally

```bash
cd 07_PCA_Feature_Reduction
jupyter notebook practical_07.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
