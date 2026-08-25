# Practical 04 — Distance Measures (Euclidean, Manhattan, Cosine)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/04_Distance_Measures/practical_04.ipynb)

**Aim:** To implement and compare the Euclidean, Manhattan and Cosine distance measures on a sample dataset.

**Dataset:** `data/wine.csv` — the first 5 wine samples (13 chemical measurements each) are used as feature vectors.

## Objectives

1. Take a few sample rows from the Wine dataset as feature vectors.
2. Compute the Euclidean, Manhattan and Cosine distance matrices using Scikit-Learn.
3. Display the three matrices as heatmaps and compare their behaviour.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the three heatmaps show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 04_Distance_Measures
jupyter notebook practical_04.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Expected result

Euclidean and Manhattan distances are dominated by the large-valued `proline` column, while cosine distance stays near zero because it compares only the direction of the vectors.

## Viva points

- Euclidean = straight-line distance; Manhattan = sum of absolute differences; Cosine = angle between vectors.
- Why features must be scaled before using Euclidean or Manhattan distance.
- Why cosine distance is the usual choice for text data.
- The diagonal of every distance matrix is zero.

`data/customers_rfm.csv`, `data/text_documents.csv` and `data/text_tfidf.csv` are kept for optional practice on other data types.
