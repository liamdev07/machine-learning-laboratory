# Practical 09 — Support Vector Machine with Hyperparameter Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/09_SVM_Hyperparameter_Tuning/practical_09.ipynb)

**Aim:** To implement a Support Vector Machine classifier and tune its hyperparameters using GridSearchCV.

**Dataset:** `data/heart_disease.csv` — 303 patients, 13 clinical features, `target` = 1 (disease) or 0 (no disease).

## Objectives

1. Build a scaling + SVM pipeline on the heart-disease dataset.
2. Search for the best combination of `C`, `gamma` and `kernel` using 5-fold cross-validation.
3. Report the test accuracy and plot the effect of `C` along with the confusion matrix.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the C curve and confusion matrix show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 09_SVM_Hyperparameter_Tuning
jupyter notebook practical_09.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Expected result

GridSearchCV selects the linear kernel with C = 1 (about 0.84 cross-validation accuracy); the test accuracy is close to 0.79 on the 61 held-out patients.

## Viva points

- SVM finds the boundary with the **maximum margin** between the two classes.
- `C` controls the penalty for misclassified points: small C = simpler model, large C = risk of overfitting.
- `gamma` controls how far the influence of one point reaches in the RBF kernel.
- `make_pipeline(StandardScaler(), SVC())` keeps scaling **inside** cross-validation and prevents leakage.
- The best cross-validation score is not a promise of the same test score — explain why.
