# Practical 05 — Decision Tree (ID3)

**Syllabus (Parul University):** *Implement a Decision Tree (ID3) on a dataset and evaluate its performance.*  
**Kernel:** `ml-lab-kernel`  
**Data:** Quinlan *Play Tennis* (categorical ID3 walkthrough) + Wisconsin Breast Cancer (sklearn entropy tree)

## Objectives

1. Compute **Shannon entropy** $H(S)\approx 0.940$ on Play Tennis and **information gain** for Outlook, Temperature, Humidity, Wind; show **Outlook** is the ID3 root.
2. Grow a **recursive ID3 tree from scratch** (Python/NumPy) and print it in **ASCII**.
3. Fit `DecisionTreeClassifier(criterion='entropy')` with `train_test_split` and a search over `max_depth` and `min_samples_split`.
4. Plot the **full tree**, **train vs test accuracy vs depth** (overfitting), and a **confusion matrix + ROC–AUC**.

## Files

| Path | Role |
|------|------|
| `practical_05.ipynb` | Full lab |
| `data/play_tennis.csv` | 14-row weather → play |
| `data/breast_cancer.csv` | Wisconsin diagnostic features + `diagnosis` |

## Swap your CSV (CONFIG cell)

```python
DATA_PATH = "data/breast_cancer.csv"
TARGET_COLUMN = "diagnosis"
FEATURE_COLUMNS = None          # or a list of numeric names
DROP_COLUMNS = []
PLAY_TENNIS_PATH = "data/play_tennis.csv"
```

## Dataset loading (fallback)

1. Local CSVs from CONFIG  
2. Colab Drive / upload  
3. Tennis: in-notebook 14-row table; cancer: `sklearn.datasets.load_breast_cancer`

## How to run

```bash
cd 05_Decision_Tree_ID3
jupyter notebook practical_05.ipynb
```

Select **Python (ML Lab Kernel)**.

## Assessment hints

- Reproduce $H(S)\approx 0.940$ and the IG ranking without looking at sklearn.
- ID3 does not reuse a discrete attribute on a path; sklearn numeric trees use binary thresholds.
- Quote **test** accuracy and ROC–AUC, not only training fit.
- A rising train curve and a falling test curve vs `max_depth` is overfitting.
