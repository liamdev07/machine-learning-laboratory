# M.Tech Machine Learning Laboratory

Syllabus-aligned **Machine Learning practicals** for postgraduate (M.Tech) coursework. Every practical is one small folder containing a clean Jupyter notebook and its dataset.

Each notebook follows the same simple **4-section classroom format**:

| Section | Content |
|---------|---------|
| 1. Aim & Objectives | What the experiment does, in 2–3 lines |
| 2. Complete Python Code | One clean, readable code cell (NumPy / Pandas / Scikit-Learn + a Matplotlib plot) |
| 3. Line-by-Line Code Explanation | What every import, function and variable means, in plain language |
| 4. Output & Graph Interpretation | Two lines explaining the result and the graph |

> **Scope note:** the `extra/` directory is **not** part of this laboratory and is left untouched.

## How to run

**Google Colab (nothing to install):** click the *Open in Colab* badge of any practical in the table below, then choose **Runtime → Run all**. The notebook reads its CSV directly from this repository, so no upload is needed.

**Local machine (standard Python 3):**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Open any `practical_XX.ipynb` and select the normal **Python 3** kernel — no custom kernel is required. The notebooks need only `numpy`, `pandas`, `matplotlib` and `scikit-learn`.

Every notebook picks its data automatically:

```python
path = CSV if os.path.exists(CSV) else URL   # local file first, GitHub copy on Colab
df = pd.read_csv(path)
```

## Laboratory index

| No. | Experiment | Dataset | Open in Colab |
|-----|------------|---------|---------------|
| 01 | [Python Basics for ML](01_Python_ML_Basics/) | California housing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/01_Python_ML_Basics/practical_01.ipynb) |
| 02 | [Covariance and Correlation](02_Covariance_Correlation/) | Iris | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/02_Covariance_Correlation/practical_02.ipynb) |
| 03 | [Linear Regression and Residual Error](03_Linear_Regression/) | Medical insurance | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/03_Linear_Regression/practical_03.ipynb) |
| 04 | [Distance Measures](04_Distance_Measures/) | Wine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/04_Distance_Measures/practical_04.ipynb) |
| 05 | [Decision Tree using ID3](05_Decision_Tree_ID3/) | Play Tennis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/05_Decision_Tree_ID3/practical_05.ipynb) |
| 06 | [K-Nearest Neighbours](06_KNN_Classifier/) | Wine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/06_KNN_Classifier/practical_06.ipynb) |
| 07 | [Feature Reduction using PCA](07_PCA_Feature_Reduction/) | Breast cancer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/07_PCA_Feature_Reduction/practical_07.ipynb) |
| 08 | [Naive Bayes for Text](08_Naive_Bayes_Text/) | SMS spam | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/08_Naive_Bayes_Text/practical_08.ipynb) |
| 09 | [SVM with Hyperparameter Tuning](09_SVM_Hyperparameter_Tuning/) | Heart disease | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/09_SVM_Hyperparameter_Tuning/practical_09.ipynb) |
| 10 | [Perceptron Learning Algorithm](10_Perceptron_Algorithm/) | Sonar | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/10_Perceptron_Algorithm/practical_10.ipynb) |

Classroom lecture notebooks are kept separately in [`theory_class_notebooks/`](theory_class_notebooks/).

## Repository layout

```text
01_Python_ML_Basics/
    practical_01.ipynb      <- the 4-section lab notebook
    data/                   <- the CSV used by that notebook
    README.md
...
10_Perceptron_Algorithm/
theory_class_notebooks/     <- lecture / demo notebooks (not graded labs)
requirements.txt
```

## Academic use

- Treat the notebooks as **lab records**: run them from top to bottom and keep the outputs for submission.
- Every result printed in a notebook comes from the code in that same notebook, so any number can be reproduced during viva.
- Cite the datasets and libraries, and follow your university academic integrity policy for the written record.
