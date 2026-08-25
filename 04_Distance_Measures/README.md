# Practical 04 — Distance Measures (Euclidean, Manhattan, Cosine)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/04_Distance_Measures/practical_04.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-pairwise-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> Distance is the engine inside KNN, k-means and every similarity search — this practical shows the three that matter most.

## Aim & Objectives

**Aim:** To implement and compare the Euclidean, Manhattan and Cosine distance measures on a sample dataset.

**Objectives**

1. Take a few sample rows from the Wine dataset as feature vectors.
2. Compute one pair of distances by hand with NumPy, then verify with Scikit-Learn.
3. Build the full Euclidean, Manhattan and Cosine distance matrices.
4. Display the three matrices as heatmaps and explain why they disagree.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/wine.csv` |
| Size | 178 rows × 14 columns (first **5** rows used as vectors) |
| Features | 13 chemical measurements (`alcohol` … `proline`) |
| Label column | `cultivar` (dropped — distance needs numbers only) |
| Extra data | `data/customers_rfm.csv`, `data/text_documents.csv`, `data/text_tfidf.csv` |

```python
CSV = "data/wine.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/04_Distance_Measures/data/wine.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → manual check → three matrices → three heatmaps |
| 3 | Line-by-Line Code Explanation | Every formula, import and function in plain language |
| 4 | Output & Graph Interpretation | Why cosine behaves differently from the other two |

## Expected outputs & results

Manual check between Wine 1 and Wine 2 (NumPy result equals the Scikit-Learn matrix entry):

| Measure | Formula | Value |
|---------|---------|------:|
| Euclidean | √Σ(aᵢ − bᵢ)² | 31.265 |
| Manhattan | Σ\|aᵢ − bᵢ\| | 51.060 |
| Cosine | 1 − (a·b)/(‖a‖‖b‖) | 0.000291 |

**Figure — three heatmaps** (Euclidean, Manhattan, Cosine) with the value written inside every cell. All three have a zero diagonal, but the Euclidean and Manhattan panels are visually dominated by the largest-valued column, `proline`.

## Viva Q&A highlights

**Q1. Define the three measures in one line each.**
Euclidean is the straight-line distance; Manhattan is the sum of absolute differences (city-block); Cosine measures the angle between two vectors, ignoring their length.

**Q2. Why is the diagonal always zero?**
Every row is compared with itself, so all coordinate differences are zero.

**Q3. Why do Euclidean and Manhattan look almost identical here?**
Both add up coordinate differences, and one feature (`proline`, in the hundreds) dwarfs the rest, so it controls both totals.

**Q4. What should be done before using them on real data?**
Standardise the features, otherwise a large-unit column decides the distance on its own.

**Q5. Why is cosine the usual choice for text?**
A long document repeats words and produces a large vector, but its topic is unchanged. Cosine compares direction, so document length does not distort similarity.

**Q6. How are cosine similarity and cosine distance related?**
Cosine distance = 1 − cosine similarity. Similarity 1 means identical direction, so the distance is 0.

## Run it locally

```bash
cd 04_Distance_Measures
jupyter notebook practical_04.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
