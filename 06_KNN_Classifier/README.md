# Practical 06 — K-Nearest Neighbours (KNN) Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/06_KNN_Classifier/practical_06.ipynb)

**Aim:** To implement the K-Nearest Neighbours classification algorithm and evaluate its accuracy.

**Dataset:** `data/wine.csv` — 178 wines, 13 chemical measurements, 3 cultivars.

## Objectives

1. Train a KNN classifier on the Wine dataset after applying feature scaling.
2. Measure accuracy and display the confusion matrix on the test data.
3. Plot accuracy against different values of k to choose the best k.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the accuracy curve and confusion matrix show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 06_KNN_Classifier
jupyter notebook practical_06.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Expected result

Accuracy is about 0.94 at k = 5 and reaches roughly 0.98 near k = 13, with almost all test wines on the diagonal of the confusion matrix.

## Viva points

- KNN is a **lazy** learner: it stores the training data and computes distances only at prediction time.
- Feature scaling is compulsory, otherwise `proline` (in hundreds) decides every neighbour.
- The scaler is fitted on the training data only, then applied to the test data.
- Small k follows noise, very large k over-smooths — hence the accuracy-vs-k curve.
