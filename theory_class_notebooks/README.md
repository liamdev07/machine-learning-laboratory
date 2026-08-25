# Theory Class Notebooks

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-FFCA28?style=flat-square)](../LICENSE)

Classroom **lecture notebooks** and **instructor demos**. These are teaching material, not graded experiments, so they do **not** follow the 4-section lab format used by practicals 01–10.

| Location | Role |
|----------|------|
| [`01_Python_ML_Basics/`](../01_Python_ML_Basics/) … [`10_Perceptron_Algorithm/`](../10_Perceptron_Algorithm/) | Official lab practicals (4-section format) |
| `theory_class_notebooks/` | Lectures, slides-as-notebooks, live demos |
| `extra/` | Personal practice work — out of scope for both |

## Lecture index

| Topic | Notebook | Launch | Description |
|-------|----------|:------:|-------------|
| Decision trees | [`decision_trees/lecture_decision_trees.ipynb`](decision_trees/lecture_decision_trees.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/theory_class_notebooks/decision_trees/lecture_decision_trees.ipynb) | Entropy vs Gini, ID3 vs CART, continuous splits, pruning, classification vs regression trees. Not Practical 05. |

## How to run

**Google Colab:** click a badge in the table, then **Runtime → Run all**.

**Local Jupyter:**

```bash
cd theory_class_notebooks
jupyter notebook
```

Every notebook here uses the standard **Python 3** kernel — no custom kernel is required. Packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Adding a lecture notebook

1. Copy the `.ipynb` (plus any small demo data) into this folder or a dated subfolder such as `2026-08-24_intro/`.
2. Add **one row** to the lecture index above: topic, relative link, Colab badge, one-line description.
3. Keep the notebook on the `python3` kernel so the Colab badge works for everyone.
4. Do not move or rewrite files under the numbered practical folders or under `extra/`.

Contributions to these lectures are welcome — see the [contribution guide in the root README](../README.md#-open-for-open-source-contributions).
