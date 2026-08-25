# Practical 05 — Decision Tree using ID3 (Entropy)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/05_Decision_Tree_ID3/practical_05.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-DecisionTree-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> The classic Play-Tennis table, entropy computed by hand, and a tree you can read like a set of if-else rules.

## Aim & Objectives

**Aim:** To implement a Decision Tree classifier using the ID3 approach (entropy and information gain) and analyse its performance.

**Objectives**

1. Load the Play-Tennis dataset and compute the dataset entropy H(S) manually.
2. Convert the categorical columns into 0/1 columns with one-hot encoding.
3. Train a `DecisionTreeClassifier` with `criterion="entropy"`, which is the ID3 splitting rule.
4. Print the tree as text rules, measure accuracy, and plot the tree diagram.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/play_tennis.csv` |
| Size | 14 rows × 5 columns |
| Features | `outlook`, `temperature`, `humidity`, `wind` (all categorical) |
| Target | `play` — 9 yes / 5 no |
| Encoding note | Read with `encoding="utf-8-sig"` to strip the byte-order mark from the first header |
| Extra data | `data/breast_cancer.csv` to repeat the experiment on numeric data |

```python
CSV = "data/play_tennis.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/05_Decision_Tree_ID3/data/play_tennis.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path, encoding="utf-8-sig")
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → entropy → one-hot → fit → rules → tree plot |
| 3 | Line-by-Line Code Explanation | The entropy formula and every Scikit-Learn call in plain language |
| 4 | Output & Graph Interpretation | Why `outlook` becomes the root and what accuracy 1.0 really means |

## Expected outputs & results

| Quantity | Value |
|----------|------:|
| Class counts | yes = 9, no = 5 |
| Dataset entropy H(S) | 0.9403 |
| Root split | `outlook_overcast` |
| Training accuracy | 1.0 |

**Outputs produced**

- A text version of the tree (`export_text`) that can be copied straight into a lab record.
- A `plot_tree` diagram where each box shows the test, its entropy, the sample count and the predicted class.

## Viva Q&A highlights

**Q1. Write the entropy formula and interpret 0.94.**
H(S) = −Σ p·log₂p. With 9 yes and 5 no, H(S) = 0.9403 — close to 1, so the labels are almost evenly mixed and the node is impure.

**Q2. How does ID3 choose the root?**
It computes the information gain (entropy before the split minus the weighted entropy after) for every feature and picks the largest. Here that is `outlook`.

**Q3. Difference between ID3 and CART?**
ID3 uses entropy / information gain and allows multiway categorical splits. CART builds binary splits and normally uses the Gini index; Scikit-Learn implements CART, so `criterion="entropy"` gives ID3-style splitting with binary branches.

**Q4. Why is one-hot encoding needed?**
Scikit-Learn accepts numbers only. `pd.get_dummies` turns `outlook=sunny` into its own 0/1 column instead of inventing a false numeric order.

**Q5. Accuracy is 1.0 — is the model excellent?**
No. The tree was tested on the same 14 rows it learned, so this is memorisation. On real data you hold out a test set and limit growth with `max_depth` or pruning.

**Q6. What is overfitting in a tree?**
Growing branches until every leaf is pure, capturing noise instead of the pattern, so accuracy collapses on unseen data.

## Run it locally

```bash
cd 05_Decision_Tree_ID3
jupyter notebook practical_05.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
