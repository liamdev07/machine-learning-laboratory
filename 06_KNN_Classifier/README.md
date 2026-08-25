# Practical 06 — K-Nearest Neighbours (KNN) Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/06_KNN_Classifier/practical_06.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-KNeighbors-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> No training equation at all — the data *is* the model. This practical also shows why scaling is not optional.

## Aim & Objectives

**Aim:** To implement the K-Nearest Neighbours classification algorithm and evaluate its accuracy.

**Objectives**

1. Split the Wine dataset and scale the features using training statistics only.
2. Train a KNN classifier with k = 5 and measure accuracy, precision, recall and F1.
3. Repeat for k = 1 … 20 and plot the accuracy curve to select the best k.
4. Read the confusion matrix to see which cultivars get mixed up.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/wine.csv` |
| Size | 178 rows × 14 columns |
| Features | 13 chemical measurements (`alcohol` … `proline`) |
| Target | `cultivar` — `class_0`, `class_1`, `class_2` |
| Split | 70% train / 30% test, stratified, `random_state=42` |
| Preprocessing | `StandardScaler` fitted on train, applied to test |

```python
CSV = "data/wine.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/06_KNN_Classifier/data/wine.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → split → scale → fit k=5 → k-sweep → two plots |
| 3 | Line-by-Line Code Explanation | Why `fit_transform` on train and `transform` on test, and every other call |
| 4 | Output & Graph Interpretation | What the accuracy curve and confusion matrix show |

## Expected outputs & results

| Quantity | Value |
|----------|------:|
| Accuracy at k = 5 | 0.9444 |
| Best k found | 13 |
| Accuracy at best k | 0.9815 |
| Weakest class | `class_2` (precision 0.83 at k = 5) |

**Figure — two panels**

- *Left:* accuracy vs k for k = 1…20, staying above 0.92 throughout, with a red dashed line at the best k.
- *Right:* confusion matrix at k = 5, almost entirely on the diagonal.

## Viva Q&A highlights

**Q1. Why is KNN called a lazy learner?**
`fit()` only stores the training data. All the work — computing distances and voting — happens at prediction time.

**Q2. What happens if you skip scaling?**
`proline` runs into the hundreds while `hue` is around 1, so `proline` alone decides which neighbours are "nearest" and the other 12 features are effectively ignored.

**Q3. Why `fit_transform` on train but only `transform` on test?**
The mean and standard deviation must come from training data alone. Fitting on test rows leaks information about unseen data into the model.

**Q4. How does the choice of k affect the result?**
Small k follows individual noisy points (low bias, high variance); very large k averages over distant points and blurs class boundaries (high bias). The curve is how we pick the middle ground.

**Q5. Why prefer an odd k?**
With two classes an odd k prevents a tied vote. With three classes ties are still possible, so Scikit-Learn breaks them by class order.

**Q6. What does `stratify=y` do in the split?**
It keeps the same proportion of the three cultivars in both the training and testing sets, which matters on a small 178-row dataset.

## Run it locally

```bash
cd 06_KNN_Classifier
jupyter notebook practical_06.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
