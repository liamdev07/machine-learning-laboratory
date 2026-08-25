# Practical 10 — Perceptron Learning Algorithm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/10_Perceptron_Algorithm/practical_10.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Perceptron-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> The original neural unit from 1958, trained one epoch at a time so you can watch it learn — and watch where it stalls.

## Aim & Objectives

**Aim:** To implement the Perceptron learning algorithm and evaluate its performance on a real dataset.

**Objectives**

1. Train a Perceptron on the Sonar dataset (mines vs rocks) after scaling the features.
2. Call `partial_fit` in a loop so accuracy can be recorded after every epoch.
3. Inspect the learned weights and bias.
4. Report the final accuracy with a confusion matrix and plot the learning curve.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/sonar.csv` |
| Size | 208 rows × 61 columns |
| Features | `f00` … `f59` — sonar energy in 60 frequency bands |
| Target | `label` — `M` mine (111) / `R` rock (97), encoded as 1 / 0 |
| Split | 70% train / 30% test, stratified, `random_state=42` |
| Training | 30 epochs of `partial_fit`, learning rate `eta0=0.01` |

```python
CSV = "data/sonar.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/10_Perceptron_Algorithm/data/sonar.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → encode → split → scale → 30-epoch loop → report → two plots |
| 3 | Line-by-Line Code Explanation | The update rule, `eta0`, epochs and `partial_fit` in plain language |
| 4 | Output & Graph Interpretation | Why the curve oscillates instead of settling |

## Expected outputs & results

| Quantity | Value |
|----------|------:|
| Final training accuracy | 0.8759 |
| Final test accuracy | 0.7937 |
| Mine precision / recall | 0.82 / 0.79 |
| Rock precision / recall | 0.77 / 0.79 |

**Figure — two panels**

- *Left:* learning curve over 30 epochs — training accuracy climbs quickly, then oscillates roughly between 0.85 and 0.91 while test accuracy hovers near 0.75–0.80.
- *Right:* confusion matrix with Rock / Mine labels, showing a few mines predicted as rocks.

## Viva Q&A highlights

**Q1. Describe the perceptron in one sentence.**
It computes the weighted sum `w·x + b` and applies a step (Heaviside) activation, predicting class 1 when the sum is positive and class 0 otherwise.

**Q2. State the update rule.**
`w = w + η (y_true − y_pred) x` and `b = b + η (y_true − y_pred)`, applied only when a sample is misclassified; correct predictions leave the weights unchanged.

**Q3. What is an epoch, and why loop `partial_fit`?**
One epoch is a single complete pass over the training data. Calling `partial_fit` in a loop trains one epoch at a time, which lets us record accuracy after each pass and draw the learning curve.

**Q4. What does the learning rate `eta0` do?**
It scales each correction. Too large and the weights jump past good solutions; too small and learning crawls.

**Q5. Why does the curve keep oscillating instead of converging?**
The perceptron convergence theorem guarantees a stopping point only for linearly separable data. Sonar is not perfectly separable, so some sample is always misclassified and keeps nudging the weights.

**Q6. Why can a single perceptron not solve XOR?**
XOR is not linearly separable — no single straight line divides its two classes. That 1969 limitation is exactly what multi-layer networks were built to overcome.

## Run it locally

```bash
cd 10_Perceptron_Algorithm
jupyter notebook practical_10.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
