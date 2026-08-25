# Practical 03 — Linear Regression and Residual Error

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/03_Linear_Regression/practical_03.ipynb)

**Aim:** To implement Simple Linear Regression using Scikit-Learn and compute the residual error.

**Dataset:** `data/medical_insurance.csv` — 1338 customers (age, sex, bmi, children, smoker, region, charges).

## Objectives

1. Fit a straight line `charges = m * age + c` on the medical insurance dataset.
2. Evaluate the model using MSE, RMSE and the R² score on unseen test data.
3. Plot the fitted regression line and the residual (error) plot.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the fit line and the residual plot show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**.

**Local machine:**

```bash
cd 03_Linear_Regression
jupyter notebook practical_03.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Expected result

The line has a positive slope of about 240, so charges rise with age, but R² is only about 0.12 — age alone cannot explain the cost.

## Viva points

- Residual = actual − predicted; MSE squares those errors, RMSE brings them back to the original unit.
- Why `X` must be 2-D (`df[["age"]]`) while `y` is 1-D.
- Why the model is fitted on the training split and scored on the test split.
- Adding `smoker` as a feature would raise R² sharply — explain why.
