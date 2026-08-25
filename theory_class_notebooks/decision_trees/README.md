# Decision Trees — Classroom Lecture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/theory_class_notebooks/decision_trees/lecture_decision_trees.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-DecisionTree-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> Lecture companion for decision trees. This is **not** Practical 05 — the graded ID3 experiment lives in [`05_Decision_Tree_ID3/`](../../05_Decision_Tree_ID3/) in the simplified 4-section format.

## What this notebook covers

- Shannon entropy vs Gini impurity, and when the two disagree
- ID3 (information gain, multiway categorical splits) vs CART (binary splits, numeric thresholds)
- Handling continuous features through threshold search
- Pruning: pre-pruning hyperparameters and cost-complexity (post) pruning
- Classification trees vs regression trees (MSE splits), visualised with `plot_tree`

## File info

| Item | Value |
|------|-------|
| Notebook | [`lecture_decision_trees.ipynb`](lecture_decision_trees.ipynb) |
| Kernel | Standard **Python 3** (Colab or local Jupyter) |
| Packages | `numpy`, `pandas`, `matplotlib`, `scikit-learn` |
| Data | Built-in Scikit-Learn datasets — nothing to download |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local Jupyter:**

```bash
cd theory_class_notebooks/decision_trees
jupyter notebook lecture_decision_trees.ipynb
```

## Related

| Resource | Link |
|----------|------|
| Graded ID3 practical | [`05_Decision_Tree_ID3/`](../../05_Decision_Tree_ID3/) |
| Lecture index | [`theory_class_notebooks/`](../) |
| Contribution guide | [root README](../../README.md#-open-for-open-source-contributions) |
