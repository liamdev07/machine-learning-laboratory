# Practical 08 — Naive Bayes Classifier for Text (SMS Spam)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/08_Naive_Bayes_Text/practical_08.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-MultinomialNB-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> Turning sentences into numbers, then letting Bayes' theorem catch the spam — about 98% of it.

## Aim & Objectives

**Aim:** To implement the Naive Bayes classifier for text classification and evaluate its performance.

**Objectives**

1. Convert SMS messages into a numeric word-count matrix using `CountVectorizer`.
2. Train a `MultinomialNB` model to label each message as ham or spam.
3. Report accuracy, precision, recall and the confusion matrix.
4. Inspect the strongest spam words and classify two brand-new messages.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/sms_spam.csv` |
| Size | 5572 rows × 2 columns |
| Columns | `label` (ham / spam), `message` (raw text) |
| Class balance | 4825 ham, 747 spam (imbalanced) |
| Split | 80% train / 20% test, stratified, `random_state=42` |
| Vectoriser | `CountVectorizer(stop_words="english")` → 7403-word vocabulary |

```python
CSV = "data/sms_spam.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/08_Naive_Bayes_Text/data/sms_spam.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → split → vectorise → fit → report → new messages → two plots |
| 3 | Line-by-Line Code Explanation | Bag of words, Laplace smoothing and every Scikit-Learn call |
| 4 | Output & Graph Interpretation | What the top-words chart and confusion matrix show |

## Expected outputs & results

| Quantity | Value |
|----------|------:|
| Vocabulary size | 7403 |
| Test accuracy | 0.9848 |
| Spam precision | 0.96 |
| Spam recall | 0.92 |
| Ham F1-score | 0.99 |

**Live predictions**

| Message | Prediction |
|---------|------------|
| "Congratulations! You have won a FREE ticket, claim your prize now" | `spam` |
| "Hey, are we still meeting for the lab practical tomorrow?" | `ham` |

**Figure — two panels**

- *Left:* the ten highest-probability spam words — "free", "txt", "ur", "stop", "text", "claim", "mobile", "reply", "www", "prize".
- *Right:* confusion matrix with very few off-diagonal entries.

## Viva Q&A highlights

**Q1. State Bayes' theorem as used here.**
P(class | words) ∝ P(class) × ∏ P(wordᵢ | class). The class with the larger value wins.

**Q2. Why is the model called *naive*?**
It assumes every word occurs independently of the others given the class. Real language breaks that assumption, yet the classifier still performs very well.

**Q3. What does `alpha=1.0` do?**
Laplace (add-one) smoothing. It adds 1 to every word count so a word never seen with a class does not force the whole probability product to zero.

**Q4. Why must `transform` — not `fit_transform` — be used on the test set?**
The vocabulary must come from the training text only. Refitting on test data would leak unseen vocabulary into the model and change the feature columns.

**Q5. Accuracy is 98% — why also report precision and recall?**
The data is imbalanced (87% ham). Predicting "ham" always would already score 87%. Spam recall of 0.92 is the number that shows real spam is being caught.

**Q6. Which is worse here, a false positive or a false negative?**
Marking a genuine SMS as spam (false positive) is usually worse, because the user may never see an important message. That is why spam precision matters.

## Run it locally

```bash
cd 08_Naive_Bayes_Text
jupyter notebook practical_08.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
