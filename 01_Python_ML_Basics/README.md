# Practical 01 — Python Basics for Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/01_Python_ML_Basics/practical_01.ipynb)

**Aim:** To study the basic Python libraries used in Machine Learning — NumPy, Pandas and Matplotlib.

**Dataset:** `data/california_housing_lab_benchmark.csv` — 400 rows, 10 columns (income, house age, rooms, population, location and house price).

## Objectives

1. Load a CSV dataset into a Pandas DataFrame and inspect its shape, head and summary statistics.
2. Perform basic NumPy operations (mean, standard deviation, maximum) on the data columns.
3. Visualise the data with a Matplotlib histogram and a scatter plot.

## Notebook structure

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | Purpose of the experiment |
| 2. Complete Python Code | One clean code cell that runs top to bottom |
| 3. Line-by-Line Code Explanation | Meaning of every import, function and variable |
| 4. Output & Graph Interpretation | What the printed values and the graphs show |

## How to run

**Google Colab:** click the badge above, then **Runtime → Run all**. The CSV is fetched from this repository automatically.

**Local machine:**

```bash
cd 01_Python_ML_Basics
jupyter notebook practical_01.ipynb
```

Use the standard **Python 3** kernel. Required packages: `numpy`, `pandas`, `matplotlib`.

## Expected result

Prices are spread in a bell shape around an average of about 2.7 lakh, and the income-vs-price scatter shows a strong upward trend (correlation ≈ 0.91).

## Viva points

- Difference between a NumPy array and a Pandas DataFrame.
- What `df.describe()` and `df.isnull().sum()` tell you before any modelling.
- Why a scatter plot is used to check the relation between two numeric columns.
