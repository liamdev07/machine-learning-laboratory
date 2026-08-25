# Practical 05 — Decision Tree using ID3 (Entropy)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/05_Decision_Tree_ID3/practical_05.ipynb)

**Aim:** To implement a Decision Tree classifier using the ID3 approach (entropy and information gain) and analyse its performance.

**Dataset:** `data/play_tennis.csv` — the classic 14-row Play Tennis table (outlook, temperature, humidity, wind → play).

## Objectives

1. Load the Play-Tennis dataset and convert its categorical columns into numbers.
2. Train a `DecisionTreeClassifier` with `criterion="entropy"`, which is the ID3 splitting rule.
3. Print the tree rules, check the accuracy, and plot the tree diagram.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the tree diagram and accuracy show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 05_Decision_Tree_ID3
jupyter notebook practical_05.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Expected result

The dataset entropy is H(S) ≈ 0.94, the tree splits first on `outlook`, and the training accuracy is 1.0 on these 14 rows.

## Viva points

- Entropy formula H(S) = −Σ p·log₂p and what the value 0.94 means.
- Information gain decides which feature becomes the root.
- ID3 uses entropy/information gain; CART uses the Gini index.
- Accuracy 1.0 on 14 rows is memorisation — mention `max_depth` pruning on larger data.

`data/breast_cancer.csv` is kept in this folder if you want to repeat the experiment on a larger numeric dataset.
