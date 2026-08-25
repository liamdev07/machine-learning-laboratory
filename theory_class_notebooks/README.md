# Theory class notebooks

Classroom **lecture notebooks** and **instructor demos** live here. They are **not** part of the ten graded M.Tech ML laboratory practicals (`01_…` through `10_…`), so they do not follow the 4-section lab format.

| Location | Role |
|----------|------|
| `01_Python_ML_Basics/` … `10_Perceptron_Algorithm/` | Official lab practicals (4-section format) |
| `theory_class_notebooks/` | Lectures, slides-as-notebooks, live demos |
| `extra/` | Out of scope — not a syllabus lab and not a lecture log |

## How to run

Every notebook here uses the standard **Python 3** kernel, so it opens directly in Google Colab or in a local Jupyter installation. No custom kernel is required — only `numpy`, `pandas`, `matplotlib` and `scikit-learn`.

```bash
cd theory_class_notebooks
jupyter notebook
```

## How to log a notebook

1. Copy the `.ipynb` (and any small demo data it needs) into this folder, or into a dated subfolder (`2026-08-24_intro/` is fine).
2. Add **one row** to the index table below (keep topic names short and link the file).
3. Do not move or rewrite files under the numbered practical folders or under `extra/`.

## Lecture index

| Topic | Notebook | Open in Colab | Description |
|-------|----------|---------------|-------------|
| Decision trees | [`decision_trees/lecture_decision_trees.ipynb`](decision_trees/lecture_decision_trees.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/theory_class_notebooks/decision_trees/lecture_decision_trees.ipynb) | Entropy vs Gini, ID3 vs CART, continuous splits, pruning, classification vs regression trees (lecture; not Practical 05). |
