# Practical 2: Compute Covariance and Correlation Matrix for a Given Dataset

---

## 1. Objective of the Practical

The main goal of this practical is to:

- Understand the concepts of **covariance** and **correlation**
- Compute the **covariance matrix** and **correlation matrix** for a real-world dataset
- Interpret how different features in a dataset are related to each other
- Build a foundation for feature analysis and data preprocessing in Machine Learning

By the end of this practical, you should be able to measure and visualize relationships between numerical variables in a dataset.

---

## 2. What is Covariance?

**Covariance** measures how two variables change together.

- If two variables **increase together**, covariance is **positive**
- If one variable **increases while the other decreases**, covariance is **negative**
- If there is **no clear relationship**, covariance is **close to zero**

### Formula

For two variables \(X\) and \(Y\) with \(n\) observations:

\[
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
\]

Where:
- \(X_i, Y_i\) = individual data points
- \(\bar{X}, \bar{Y}\) = mean of X and Y
- \(n\) = number of observations

### Key Points

- Covariance tells us the **direction** of the relationship (positive or negative)
- It does **not** tell us the **strength** of the relationship in a standardized way
- The value of covariance depends on the **units** of the variables (e.g., cm, kg)

---

## 3. What is Covariance Matrix?

A **covariance matrix** is a square table that shows the covariance between **every pair of numerical features** in a dataset.

For a dataset with features \(X_1, X_2, \ldots, X_n\), the covariance matrix looks like:

\[
\begin{bmatrix}
\text{Cov}(X_1, X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
\text{Cov}(X_2, X_1) & \text{Cov}(X_2, X_2) & \cdots & \text{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Cov}(X_n, X_n)
\end{bmatrix}
\]

### Important Properties

- **Diagonal elements** represent the **variance** of each feature (covariance of a variable with itself)
- The matrix is **symmetric** — Cov(X, Y) = Cov(Y, X)
- Off-diagonal elements show how pairs of features vary together

---

## 4. What is Correlation?

**Correlation** is a standardized measure of the relationship between two variables. Unlike covariance, correlation always lies between **-1 and +1**.

| Correlation Value | Meaning |
|-------------------|---------|
| **+1** | Perfect positive linear relationship |
| **0** | No linear relationship |
| **-1** | Perfect negative linear relationship |

### Formula (Pearson Correlation Coefficient)

\[
r_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
\]

Where:
- \(\sigma_X, \sigma_Y\) = standard deviations of X and Y

### Key Points

- Correlation is **unitless** — it does not depend on the scale of the variables
- It measures the **strength and direction** of a **linear** relationship
- Values closer to +1 or -1 indicate a stronger relationship

---

## 5. What is Correlation Matrix?

A **correlation matrix** is similar to a covariance matrix, but instead of covariance values, it contains **correlation coefficients** between all pairs of numerical features.

Example for 4 features:

\[
\begin{bmatrix}
1 & r_{12} & r_{13} & r_{14} \\
r_{21} & 1 & r_{23} & r_{24} \\
r_{31} & r_{32} & 1 & r_{34} \\
r_{41} & r_{42} & r_{43} & 1
\end{bmatrix}
\]

### Important Properties

- **Diagonal values are always 1** (a variable is perfectly correlated with itself)
- The matrix is **symmetric**
- Values range from **-1 to +1**, making comparison across features easy

---

## 6. Difference between Covariance and Correlation

| Aspect | Covariance | Correlation |
|--------|------------|-------------|
| **Range** | Unbounded (can be any value) | Between -1 and +1 |
| **Units** | Depends on variable units | Unitless (standardized) |
| **Interpretation** | Shows direction only | Shows both direction and strength |
| **Comparison** | Hard to compare across different feature pairs | Easy to compare across features |
| **Scale sensitivity** | Affected by scale of data | Not affected by scale |

**In simple terms:** Covariance tells us *if* two variables move together, while correlation tells us *how strongly* they move together in a comparable way.

---

## 7. Why Do We Use These Matrices in Machine Learning?

Covariance and correlation matrices are widely used in ML and data science for several reasons:

- **Feature Selection** — Highly correlated features may carry redundant information. We can remove one to reduce dimensionality.
- **Multicollinearity Detection** — In regression models, strong correlations between input features can cause unstable predictions.
- **Dimensionality Reduction** — Techniques like **PCA (Principal Component Analysis)** use the covariance matrix to find important directions in data.
- **Exploratory Data Analysis (EDA)** — Helps us understand data before building models.
- **Feature Engineering** — Identifying related features helps in creating better new features.
- **Visualization** — Correlation heatmaps make it easy to spot patterns in large datasets.

---

## 8. Steps to Be Performed in This Practical

1. **Load the dataset** — Import the Iris dataset using Python (pandas).
2. **Explore the data** — Display basic information such as shape, columns, and data types.
3. **Select numerical features** — Choose only the numerical columns (exclude ID and categorical labels).
4. **Compute the Covariance Matrix** — Use `pandas.DataFrame.cov()` or NumPy to calculate covariance between all feature pairs.
5. **Compute the Correlation Matrix** — Use `pandas.DataFrame.corr()` to calculate Pearson correlation coefficients.
6. **Display the results** — Print both matrices in a readable format.
7. **Visualize (optional)** — Plot a heatmap using `seaborn.heatmap()` to visualize correlations.
8. **Interpret the results** — Analyze which features are strongly or weakly related.

### Sample Python Code Snippet

```python
import pandas as pd

# Load dataset
df = pd.read_csv("Iris.csv")

# Select numerical columns
numerical_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
data = df[numerical_cols]

# Covariance Matrix
cov_matrix = data.cov()
print("Covariance Matrix:\n", cov_matrix)

# Correlation Matrix
corr_matrix = data.corr()
print("\nCorrelation Matrix:\n", corr_matrix)
```

---

## 9. Dataset Used (Iris Dataset)

The **Iris dataset** is one of the most famous datasets in machine learning and statistics.

### About the Dataset

- **Total samples:** 150
- **Number of features:** 4 numerical features + 1 categorical label
- **Number of classes:** 3 species of iris flowers

### Features

| Column | Description |
|--------|-------------|
| `Id` | Unique identifier for each record |
| `SepalLengthCm` | Length of the sepal (in cm) |
| `SepalWidthCm` | Width of the sepal (in cm) |
| `PetalLengthCm` | Length of the petal (in cm) |
| `PetalWidthCm` | Width of the petal (in cm) |
| `Species` | Flower species (Iris-setosa, Iris-versicolor, Iris-virginica) |

### Why Iris Dataset?

- Small and easy to understand
- Contains multiple numerical features suitable for covariance and correlation analysis
- Commonly used in academic labs for learning statistical concepts
- Helps visualize how flower measurements relate to each other

### Expected Observations

- **Petal Length** and **Petal Width** are usually **strongly positively correlated**
- **Sepal Length** may show moderate correlation with petal measurements
- **Sepal Width** often has weaker correlations with other features

---

## 10. Streamlit Web Application

This project includes a **polished Streamlit-based web interface** (`app.py`) in addition to the command-line script (`main.py`). The app provides a modern, professional dashboard for uploading datasets and analyzing covariance and correlation relationships through an interactive browser UI.

### Features

- **Professional dashboard layout** with custom styling, clear section headers, and improved visual hierarchy
- **Sidebar guidance** with app description, step-by-step instructions, and analysis notes
- **CSV file upload** with success and empty-state messages
- **Dataset overview metrics** showing row count, total columns, and analysis feature count
- **Interactive data preview** showing the first 10 rows of the uploaded dataset
- **Smart numerical feature detection** with automatic exclusion of identifier columns such as `Id`
- **Styled Covariance and Correlation matrices** with color gradients for easier interpretation
- **Tabbed statistical analysis section** for matrices, heatmap, and observations
- **Interactive Correlation Heatmap** with annotated values and colorbar
- **Automated observations** for strongest positive/negative correlations, weak relationships, and ML implications

### UI Highlights

- Identifier columns like `Id` are excluded from all statistical analysis
- Feature chips, metric cards, and section dividers improve readability
- Matrices use color-coded styling (`Blues` for covariance, `coolwarm` for correlation)
- Observations are displayed using structured info and success message boxes

### Prerequisites

Install all project dependencies (including Streamlit):

```bash
python -m pip install -r requirements.txt
```

### Run the Streamlit App

From the `lab-02` folder, run:

```bash
streamlit run app.py
```

Streamlit will start a local development server and open the app in your default web browser. If it does not open automatically, use the local URL shown in the terminal (typically `http://localhost:8501`).

### Run the Command-Line Script

To run the Python script directly:

```bash
python main.py
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| Covariance | Measures how two variables vary together (direction) |
| Covariance Matrix | Shows pairwise covariance for all features |
| Correlation | Standardized measure of linear relationship (-1 to +1) |
| Correlation Matrix | Shows pairwise correlation for all features |

Understanding these matrices is an essential step in exploratory data analysis and forms the basis for many advanced machine learning techniques.

---

**Course:** M.Tech AI & Data Science  
**Practical:** Lab 02 — Covariance and Correlation Matrix
