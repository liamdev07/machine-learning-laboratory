# Decision trees (classroom lecture)

Instructor / theory notebook for **decision trees**. This is **not** Practical 05 (`05_Decision_Tree_ID3/`). The lab remains the graded ID3 + sklearn experiment; this file is for lecture pacing and demos.

| Item | Path |
|------|------|
| Lecture notebook | [`lecture_decision_trees.ipynb`](lecture_decision_trees.ipynb) |
| Kernel | `ml-lab-kernel` (**Python (ML Lab Kernel)**) |

## Topics

- Shannon entropy vs Gini impurity
- ID3 (information gain, multiway categorical splits) vs CART (binary, Gini or entropy, numeric cuts)
- Continuous features (threshold search)
- Pruning (pre-pruning hyperparameters; cost-complexity / post-pruning)
- Classification trees vs regression trees (MSE splits) with `plot_tree`

```bash
cd theory_class_notebooks/decision_trees
jupyter notebook lecture_decision_trees.ipynb
```
