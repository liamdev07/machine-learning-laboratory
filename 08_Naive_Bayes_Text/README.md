# Practical 08 — Naive Bayes Classifier for Text (SMS Spam)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/08_Naive_Bayes_Text/practical_08.ipynb)

**Aim:** To implement the Naive Bayes classifier for text classification and evaluate its performance.

**Dataset:** `data/sms_spam.csv` — 5572 SMS messages labelled `ham` (4825) or `spam` (747).

## Objectives

1. Convert SMS messages into a numeric word-count matrix using `CountVectorizer`.
2. Train a `MultinomialNB` model to classify each message as ham or spam.
3. Report accuracy, plot the confusion matrix and test the model on new messages.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the top-words chart and confusion matrix show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 08_Naive_Bayes_Text
jupyter notebook practical_08.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Expected result

Test accuracy is about 98%, the strongest spam words are "free", "txt", "claim" and "prize", and the two new sample messages are classified correctly.

## Viva points

- Bayes theorem and why the model is called **naive** (all words assumed independent).
- `CountVectorizer` builds the vocabulary from the training text only.
- Laplace smoothing (`alpha=1`) stops an unseen word from making the probability zero.
- Accuracy alone is misleading on imbalanced data — quote spam precision and recall.
