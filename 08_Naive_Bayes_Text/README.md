# Practical 08 — Naïve Bayes for Text Classification

**Syllabus (Parul University):** *Implement a Naïve Bayes classifier for text classification.*  
**Kernel:** `ml-lab-kernel`  
**Data:** SMS Spam Collection — `data/sms_spam.csv` (label ∈ {ham, spam}, free-text `message`; public dump has 5,572 rows)

## Objectives

1. State Bayes’ rule, the **naïve** independence assumption, **multinomial** likelihoods, and **Laplace** smoothing.
2. Clean SMS text (regex), drop stopwords, tokenize; compare **Count** vs **TF–IDF** features.
3. Implement `ScratchMultinomialNB` and match `sklearn.naive_bayes.MultinomialNB`.
4. Tune `alpha` in a `TfidfVectorizer` + `MultinomialNB` pipeline.
5. Confusion matrix, ROC/PR, top tokens for spam vs ham.
6. Classify **new** SMS strings with the fitted pipeline.

## Dual load path

1. `./data/sms_spam.csv`
2. GitHub raw TSV (`label`, `message`) used in many tutorials
3. Embedded mini-corpus only if both fail (not a substitute for the full lab)

## Citation

Almeida, Hidalgo, Yamakami — SMS Spam Collection (UCI / OpenML). Almeida & Hidalgo.

## How to run

```bash
cd 08_Naive_Bayes_Text
jupyter notebook practical_08.ipynb
```
