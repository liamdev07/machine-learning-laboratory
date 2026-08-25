# Practical 03 — Linear Regression and Residual Error

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/03_Linear_Regression/practical_03.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-LinearRegression-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

> The first supervised model of the course: fit a straight line, then be honest about how wrong it is.

## Aim & Objectives

**Aim:** To implement Simple Linear Regression using Scikit-Learn and compute the residual error.

**Objectives**

1. Fit the line `charges = m × age + c` on the medical insurance dataset.
2. Split the data so the model is scored on rows it has never seen.
3. Evaluate with MSE, RMSE and the R² score.
4. Plot the fitted line and the residual plot, and read what they say about the model.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/medical_insurance.csv` |
| Size | 1338 rows × 7 columns |
| Columns | `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges` |
| Input (X) | `age` (kept 2-D as `df[["age"]]`) |
| Target (y) | `charges` |
| Split | 80% train / 20% test, `random_state=42` |

```python
CSV = "data/medical_insurance.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/03_Linear_Regression/data/medical_insurance.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → split → fit → metrics → two plots |
| 3 | Line-by-Line Code Explanation | Every import, function and variable in plain language |
| 4 | Output & Graph Interpretation | What the fitted line and the residual spread mean |

## Expected outputs & results

| Quantity | Value |
|----------|------:|
| Slope *m* | 240.597 |
| Intercept *c* | 3876.929 |
| MSE | 135,983,957.48 |
| RMSE | 11,661.22 |
| R² | 0.1241 |

**Figure — two panels**

- *Left:* test points with the red fitted line rising with age.
- *Right:* residual plot around the zero line; the errors form wide bands instead of a tight cloud.

The low R² is the teaching point: age alone explains only about 12% of the variation in charges.

## Viva Q&A highlights

**Q1. What exactly is a residual?**
Residual = actual − predicted, computed for each test row. It is the error the model makes on that customer.

**Q2. MSE, RMSE, R² — how do they differ?**
MSE averages the squared errors (units squared). RMSE is its square root, back in the original currency. R² reports the fraction of variance explained; 1.0 is perfect and 0.0 equals predicting the mean.

**Q3. Why must `X` be `df[["age"]]` and not `df["age"]`?**
Scikit-Learn expects a 2-D array of shape (n_samples, n_features). Single brackets give a 1-D Series and raise an error.

**Q4. Why is the model fitted on the training split only?**
Testing on the rows used for fitting would flatter the model. The held-out split estimates performance on new customers.

**Q5. R² is only 0.12 — is the code wrong?**
No. The model is honest; age alone is a weak predictor. Adding `smoker` (after encoding) raises R² sharply because smoking dominates insurance cost.

**Q6. What would a good residual plot look like?**
Points scattered randomly and symmetrically around zero with no funnel or curve — that would suggest the linear form and constant variance are reasonable.

## Run it locally

```bash
cd 03_Linear_Regression
jupyter notebook practical_03.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
