# Practical 01 — Python Basics for Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/01_Python_ML_Basics/practical_01.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/NumPy%20%7C%20Pandas%20%7C%20Matplotlib-013243?style=flat-square)](https://numpy.org/)

> The starting experiment of the laboratory: get comfortable with the three libraries every later practical depends on.

## Aim & Objectives

**Aim:** To study the basic Python libraries used in Machine Learning — NumPy, Pandas and Matplotlib.

**Objectives**

1. Load a CSV dataset into a Pandas DataFrame and inspect its shape, head and summary statistics.
2. Perform basic NumPy operations (mean, standard deviation, maximum, minimum) on the data columns.
3. Check for missing values before any modelling is attempted.
4. Visualise the data with a Matplotlib histogram and a scatter plot.

## Dataset Info & Loading

| Property | Value |
|----------|-------|
| File | `data/california_housing_lab_benchmark.csv` |
| Size | 400 rows × 10 columns |
| Features | `median_income`, `house_age`, `avg_rooms`, `avg_bedrooms`, `population`, `avg_occupancy`, `latitude`, `longitude`, `ocean_proximity` |
| Target of interest | `median_house_value` |
| Missing values | None |

The notebook picks its source automatically, so the same cell works locally and on Colab:

```python
CSV = "data/california_housing_lab_benchmark.csv"
URL = "https://raw.githubusercontent.com/liamdev07/machine-learning-laboratory/main/01_Python_ML_Basics/data/california_housing_lab_benchmark.csv"
path = CSV if os.path.exists(CSV) else URL
df = pd.read_csv(path)
```

## Notebook structure (4 sections)

| # | Section | What you will find |
|:-:|---------|--------------------|
| 1 | Aim & Objectives | Purpose of the experiment in three lines |
| 2 | Complete Python Code | One clean cell: load → inspect → NumPy stats → two plots |
| 3 | Line-by-Line Code Explanation | Every import, function and variable in plain language |
| 4 | Output & Graph Interpretation | What the histogram and the scatter plot prove |

## Expected outputs & results

| Output | Value |
|--------|-------|
| Dataset shape | `(400, 10)` |
| Mean median income | ≈ 4.19 |
| Mean house value | ≈ 2,70,740 |
| Income ↔ price correlation | ≈ 0.91 |

**Figure — two panels**

- *Left:* histogram of `median_house_value`, roughly bell-shaped around the average with very few extreme houses.
- *Right:* scatter of `median_income` vs `median_house_value`, showing a clear upward trend.

## Viva Q&A highlights

**Q1. What is the difference between a NumPy array and a Pandas DataFrame?**
An array holds one data type in a grid and is fast for maths; a DataFrame is a labelled table that can mix numbers and text, and it supports column names.

**Q2. Why do we call `df.describe()` before modelling?**
It shows count, mean, standard deviation, minimum and maximum together, so unrealistic ranges and scale differences between columns become visible immediately.

**Q3. What does `df.isnull().sum()` tell you?**
The number of empty cells per column. Missing values must be filled or dropped before most Scikit-Learn models will run.

**Q4. Why a histogram for price and a scatter for income vs price?**
A histogram describes the spread of a single variable; a scatter plot shows the relationship between two variables.

**Q5. Does the strong 0.91 correlation prove that income causes price?**
No. Correlation shows association only; other factors such as location and house size also drive price.

## Run it locally

```bash
cd 01_Python_ML_Basics
jupyter notebook practical_01.ipynb
```

Kernel: standard **Python 3**. Requires `numpy`, `pandas`, `matplotlib`.
