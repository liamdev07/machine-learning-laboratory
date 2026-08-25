# Practical 04 — Distance Measures

**University mapping:** *Implement different distance measures (Euclidean, Manhattan, Cosine) on sample datasets.*  
**Duration:** 1 lab session  
**Stack:** NumPy (broadcast pairwise), SciPy `cdist`, scikit-learn `pairwise_distances` / `cosine_similarity`  
**Kernel:** `ml-lab-kernel`

## Objectives

1. Compute **Manhattan $\ell_1$**, **Euclidean $\ell_2$**, **Chebyshev $\ell_\infty$**, the **dot product**, **vector norms**, **cosine similarity**, and **cosine distance** by hand on two vectors; match NumPy / SciPy / sklearn.
2. Place them in the **Minkowski $\ell_p$** family and sketch **unit balls** (diamond / circle / square).
3. Explain **distance concentration** (curse of dimensionality) with a contrast-vs-$d$ curve.
4. Implement **vectorized** pairwise matrices and cross-check `cdist` and `pairwise_distances`.
5. Contrast **scaled Wine chemistry** with **TF–IDF text**: Euclidean grows with document length; cosine does not.

## Files

| Path | Role |
|------|------|
| `practical_04.ipynb` | Full experiment |
| `data/wine.csv` | UCI Wine (assays + cultivar) |
| `data/text_tfidf.csv` | TF–IDF rows for six short documents |
| `data/text_documents.csv` | Raw text (rebuild TF–IDF if needed) |
| `data/customers_rfm.csv` | Optional extra table (not required by the notebook) |

## Swap your CSV (CONFIG cell)

Edit **only** the **DATASET CONFIG** block at the top:

```python
DATA_PATH = "data/wine.csv"
FEATURE_COLUMNS = None          # or a list of numeric column names
SAMPLE_INDEX_A = 0
SAMPLE_INDEX_B = 10
DROP_COLUMNS = ["cultivar"]
TEXT_PATH = "data/text_tfidf.csv"
```

## Dataset loading (ordered fallback)

1. Local `DATA_PATH` / `TEXT_PATH`
2. Colab Drive / upload
3. Wine: `sklearn.datasets.load_wine`; text: rebuild TF–IDF from `text_documents.csv` or an inline corpus

## How to run

```bash
cd 04_Distance_Measures
jupyter notebook practical_04.ipynb
```

## Assessment hints

- Cosine **similarity** is not a metric; sklearn cosine **distance** is $1-\cos\theta$.
- Standardise Wine before Euclidean — `proline` otherwise dominates.
- Two documents with the same term mix and different length: Euclidean large, cosine $=1$.
- In high $d$, $(d_{\max}-d_{\min})/d_{\min}$ collapses: nearest and farthest look alike.
