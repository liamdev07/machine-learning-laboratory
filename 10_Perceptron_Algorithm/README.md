# Practical 10 — Perceptron Learning Algorithm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/10_Perceptron_Algorithm/practical_10.ipynb)

**Aim:** To implement the Perceptron learning algorithm and evaluate its performance on a real dataset.

**Dataset:** `data/sonar.csv` — 208 sonar returns, 60 energy bands, label `M` (mine) or `R` (rock).

## Objectives

1. Train a Perceptron on the Sonar dataset after scaling the features.
2. Record the accuracy after every epoch to observe how learning progresses.
3. Report the final accuracy with a confusion matrix and plot the learning curve.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the learning curve and confusion matrix show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 10_Perceptron_Algorithm
jupyter notebook practical_10.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Expected result

Training accuracy climbs to roughly 0.88 and oscillates, while test accuracy settles near 0.79 — the Sonar data is not perfectly linearly separable.

## Viva points

- The perceptron computes `w·x + b` and applies a step (Heaviside) activation.
- Update rule `w = w + eta * (y_true - y_pred) * x` is applied only for misclassified samples.
- `eta0` is the learning rate; one **epoch** is one full pass over the training data.
- A single perceptron can learn AND and OR but not XOR, because XOR is not linearly separable.
- The curve keeps oscillating instead of converging when the data is not linearly separable.
