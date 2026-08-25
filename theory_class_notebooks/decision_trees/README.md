# Decision trees (classroom lecture)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/theory_class_notebooks/decision_trees/lecture_decision_trees.ipynb)

Instructor / theory notebook for **decision trees**. This is **not** Practical 05 (`05_Decision_Tree_ID3/`). The lab remains the graded ID3 experiment in the simplified 4-section format; this file is a longer lecture with extra demos.

| Item | Value |
|------|-------|
| Lecture notebook | [`lecture_decision_trees.ipynb`](lecture_decision_trees.ipynb) |
| Kernel | Standard **Python 3** (works in Colab and local Jupyter) |
| Packages | `numpy`, `pandas`, `matplotlib`, `scikit-learn` |

## Topics

- Shannon entropy vs Gini impurity
- ID3 (information gain, multiway categorical splits) vs CART (binary splits, Gini or entropy, numeric cuts)
- Continuous features (threshold search)
- Pruning (pre-pruning hyperparameters; cost-complexity / post-pruning)
- Classification trees vs regression trees (MSE splits) with `plot_tree`

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd theory_class_notebooks/decision_trees
jupyter notebook lecture_decision_trees.ipynb
```
