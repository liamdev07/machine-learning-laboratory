# M.Tech Machine Learning Laboratory

Industry-standard, syllabus-aligned **Machine Learning practicals** for postgraduate (M.Tech) coursework. Each experiment is a self-contained folder with a production-grade Jupyter notebook, local sample data, dual load paths (local disk **and** Google Colab / URL fallback), micro-level theory (including LaTeX), and viva-voce Q&A.

> **Scope note:** The existing `extra/` directory is **out of scope** for this laboratory tree. Do not treat it as part of the graded practical sequence.

## Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name ml-lab-kernel --display-name "Python (ML Lab Kernel)"
jupyter notebook
```

In Jupyter / VS Code / Cursor, select kernel **Python (ML Lab Kernel)** (`ml-lab-kernel`) so every practical uses this `.venv`.
```

**Google Colab:** upload a practical folder (or clone the repository) and run the notebook. Dataset loaders fall back to a public URL if `./data/` is not present.

**Reproducibility:** notebooks fix `random_state` / NumPy RNG seeds. Record Python and package versions in the environment cell.

## Laboratory syllabus index

| No. | Folder | Experiment | Core outcomes |
|-----|--------|------------|----------------|
| 01 | [`01_Python_ML_Basics/`](01_Python_ML_Basics/) | Python scientific stack for ML | NumPy vectorization & linear algebra, Pandas EDA, Matplotlib/Seaborn, sklearn train/test split, scaling, baseline model |
| 02 | [`02_Covariance_Correlation/`](02_Covariance_Correlation/) | Covariance, Pearson \(r\), \(\Sigma\) | Hand calculation (\(n-1\)), vectorized NumPy/Pandas, heatmaps/pairplots, signed association |
| 03 | [`03_Linear_Regression/`](03_Linear_Regression/) | Linear regression & residual error | OLS normal equations, GD from scratch, sklearn LR, MAE/MSE/RMSE/\(R^2\), residual diagnostics |
| 04 | [`04_Distance_Measures/`](04_Distance_Measures/) | Distance measures | Euclidean, Manhattan, Minkowski, Chebyshev, cosine; NumPy vs SciPy/sklearn; curse of dimensionality |
| 05 | [`05_Decision_Tree_ID3/`](05_Decision_Tree_ID3/) | Decision tree (ID3) | Entropy/IG from scratch, sklearn entropy trees, plot_tree, confusion/F1/ROC, depth vs overfit |
| 06 | [`06_KNN_Classifier/`](06_KNN_Classifier/) | k-NN classifier | Vectorized NumPy kNN, sklearn pipeline + scaling, error vs \(k\), decision boundaries, accuracy |
| 07 | [`07_PCA_Feature_Reduction/`](07_PCA_Feature_Reduction/) | PCA feature reduction | NumPy eig/SVD PCA, sklearn inverse reconstruct, scree/2D/3D/biplot, classifier before vs after |
| 08 | [`08_Naive_Bayes_Text/`](08_Naive_Bayes_Text/) | Naïve Bayes text classification | Bayes + Laplace, Count/TF–IDF, scratch MultinomialNB, sklearn pipeline, SMS spam diagnostics |
| 09 | [`09_SVM_Hyperparameter_Tuning/`](09_SVM_Hyperparameter_Tuning/) | SVM + hyperparameter tuning | Max-margin / kernels / \(C,\gamma\), NumPy dual \(f(x)\), GridSearchCV, SV plots, ROC |
| 10 | [`10_Perceptron_Algorithm/`](10_Perceptron_Algorithm/) | Perceptron learning | Heaviside, delta rule, Novikoff, AND/OR vs XOR, sklearn Perceptron on Sonar |
| 11 | `11_Data_Preprocessing/` | Data cleaning & feature engineering | Missing values, encoding, outliers, pipelines, leakage-safe transforms |
| 12 | `12_Regularized_Regression/` | Ridge / Lasso / ElasticNet | Shrinkage, residual diagnostics, RMSE, \(R^2\) |
| 13 | `13_Model_Selection/` | Validation & tuning | k-fold CV, nested CV, GridSearch/RandomizedSearch, learning curves |
| 14 | `14_Ensemble_Methods/` | Bagging, boosting, stacking | Random Forest, Gradient Boosting / XGBoost-style ideas, feature importance |
| 15 | `15_Unsupervised_Clustering/` | Clustering | k-means, hierarchical, DBSCAN, silhouette & cluster validity |
| 16 | `16_Neural_Networks/` | MLP / intro DL | Backprop intuition, sklearn MLP or a small Keras/PyTorch lab |
| 17 | `17_NLP_Fundamentals/` | Text representation | Bag-of-words, TF–IDF, simple classifiers on text |
| 18 | `18_Time_Series/` | Forecasting basics | Stationarity, lag features, walk-forward validation |
| 19 | `19_MLOps_Deployment/` | Production hygiene | Serialization, inference API sketch, metrics logging, reproducibility checklist |

This repository currently ships **Practicals 01–10** in full. Folders **11–19** are reserved.

## Practical 01 quick start

```bash
cd 01_Python_ML_Basics
jupyter notebook practical_01.ipynb
```

Local benchmark CSV: `01_Python_ML_Basics/data/california_housing_lab_benchmark.csv`.

## Practical 02 quick start

```bash
cd 02_Covariance_Correlation
jupyter notebook practical_02.ipynb
```

Local benchmarks: `data/iris.csv`, `data/auto_mpg.csv`. Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Practical 03 quick start

```bash
cd 03_Linear_Regression
jupyter notebook practical_03.ipynb
```

Local benchmark: `data/medical_insurance.csv` (medical charges). Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Practical 04 quick start

```bash
cd 04_Distance_Measures
jupyter notebook practical_04.ipynb
```

Local benchmarks: `data/wine.csv`, `data/customers_rfm.csv`, `data/text_tfidf.csv`. Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Practical 05 quick start

```bash
cd 05_Decision_Tree_ID3
jupyter notebook practical_05.ipynb
```

Local data: `data/play_tennis.csv`, `data/breast_cancer.csv`. Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Practical 06 quick start

```bash
cd 06_KNN_Classifier
jupyter notebook practical_06.ipynb
```

Local benchmark: `data/wine.csv`. Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Practical 07 quick start

```bash
cd 07_PCA_Feature_Reduction
jupyter notebook practical_07.ipynb
```

Local benchmark: `data/breast_cancer.csv`. Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Practical 08 quick start

```bash
cd 08_Naive_Bayes_Text
jupyter notebook practical_08.ipynb
```

Local benchmark: `data/sms_spam.csv` (SMS Spam Collection). Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Practical 09 quick start

```bash
cd 09_SVM_Hyperparameter_Tuning
jupyter notebook practical_09.ipynb
```

Local benchmark: `data/heart_disease.csv`. Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Practical 10 quick start

```bash
cd 10_Perceptron_Algorithm
jupyter notebook practical_10.ipynb
```

Local benchmark: `data/sonar.csv` (Mines vs Rocks). Kernel: **Python (ML Lab Kernel)** (`ml-lab-kernel`).

## Academic use

- Treat notebooks as **lab records**: run top-to-bottom, keep outputs for submission if required.
- Cite datasets and libraries. The Colab/URL fallback uses the public California Housing file from the Hands-On ML companion repository when local data is absent.
- Viva questions at the end of each notebook are indicative, not exhaustive.

## License / coursework

Use according to your university academic integrity policy. Do not submit unmodified notebooks as solely your own work if your department requires original write-ups.
