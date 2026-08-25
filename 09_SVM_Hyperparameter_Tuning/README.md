# Practical 09 — Support Vector Machine with Hyperparameter Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/09_SVM_Hyperparameter_Tuning/practical_09.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVC%20%2B%20GridSearchCV-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> Maximum-margin classification plus the honest version of tuning: cross-validation inside a pipeline, never on the test set.

## Aim & Objectives

**Aim:** To implement a Support Vector Machine classifier and tune its hyperparameters using GridSearchCV.

**Objectives**

1. Build a `StandardScaler` + `SVC` pipeline on the heart-disease dataset.
2. Observe how the penalty parameter `C` changes the cross-validation score.
3. Search `C`, `gamma` and `kernel` together with 5-fold `GridSearchCV`.
4. Report the test accuracy of the tuned model and read its confusion matrix.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/heart_disease.csv` |
| Size | 303 rows × 14 columns |
| Features | `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal` |
| Target | `target` — 1 = disease present (165), 0 = absent (138) |
| Split | 80% train / 20% test, stratified, `random_state=42` |
| Search grid | `C` ∈ {0.1, 1, 10, 100}, `gamma` ∈ {scale, 0.01, 0.1, 1}, `kernel` ∈ {linear, rbf} |

```python
CSV = "data/heart_disease.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/09_SVM_Hyperparameter_Tuning/data/heart_disease.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → split → baseline → C sweep → GridSearchCV → test score → two plots |
| 3 | Line-by-Line Code Explanation | `C`, `gamma`, kernels, pipelines and cross-validation in plain language |
| 4 | Output & Graph Interpretation | Why the C curve peaks and why the test score sits below the CV score |

## Expected outputs & results

| Quantity | Value |
|----------|------:|
| Default SVM test accuracy | 0.8197 |
| Best parameters | `C = 1`, `gamma = scale`, `kernel = linear` |
| Best cross-validation accuracy | 0.8393 |
| Tuned SVM test accuracy | 0.7869 |

Cross-validation accuracy against `C` (RBF kernel): 0.545 → **0.823** → 0.802 → 0.781 → 0.777 for C = 0.01, 0.1, 1, 10, 100.

**Figure — two panels**

- *Left:* CV accuracy vs `C` on a logarithmic x-axis — very low at C = 0.01, a peak near C = 0.1, then a slow decline.
- *Right:* confusion matrix of the tuned model on the 61 test patients.

## Viva Q&A highlights

**Q1. What does an SVM actually optimise?**
The widest possible margin between the two classes; only the closest points, the support vectors, determine the boundary.

**Q2. What is the role of `C`?**
It is the penalty for misclassified points. Small `C` tolerates errors and gives a wide, simple margin; large `C` forces the training points to be correct and risks overfitting.

**Q3. What does `gamma` control?**
How far the influence of one training point reaches in the RBF kernel. Large `gamma` gives a tight, wiggly boundary; small `gamma` gives a smooth one.

**Q4. Why wrap the scaler and the model in a pipeline?**
Cross-validation refits the pipeline on each fold, so scaling is computed from that fold's training part only. Scaling before the split would leak information.

**Q5. Why is the tuned test accuracy (0.79) lower than the CV accuracy (0.84)?**
GridSearchCV reports the best score over many combinations, which is optimistic, and the test set holds only 61 patients, so a handful of cases moves the number several points. It is not a bug.

**Q6. What does `cv=5` mean?**
The training data is split into 5 parts; the model trains on 4 and validates on 1, five times, and the scores are averaged.

## Run it locally

```bash
cd 09_SVM_Hyperparameter_Tuning
jupyter notebook practical_09.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
