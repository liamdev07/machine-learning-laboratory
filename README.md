<h1 align="center">M.Tech Machine Learning Laboratory</h1>

<p align="center">
  Ten syllabus-aligned Machine Learning practicals, written in a clean classroom format that runs anywhere — one click on Colab, or a plain <code>python3</code> kernel on your own machine.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://jupyter.org/"><img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white"></a>
  <a href="https://scikit-learn.org/"><img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"></a>
  <a href="#-open-for-open-source-contributions"><img alt="Open Source" src="https://img.shields.io/badge/Open%20Source-Yes-3DA639?style=flat-square&logo=opensourceinitiative&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-FFCA28?style=flat-square"></a>
  <a href="#-open-for-open-source-contributions"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square"></a>
  <a href="https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/01_Python_ML_Basics/practical_01.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

---

## Overview

Every practical lives in its own folder with a notebook, its dataset and a README. There is **no custom kernel, no setup cell and no hidden helper file** — open the notebook, run all cells, read the result.

Each notebook follows the same **4-section classroom format**:

| Section | Content |
|---------|---------|
| **1. Aim & Objectives** | What the experiment does, in 2–3 lines |
| **2. Complete Python Code** | One clean, readable cell (NumPy / Pandas / Scikit-Learn + a Matplotlib plot) |
| **3. Line-by-Line Code Explanation** | What every import, function and variable means, in plain language |
| **4. Output & Graph Interpretation** | Two lines explaining the result and the graph |

> **Scope note:** the `extra/` directory is personal practice work, kept out of this laboratory sequence and left untouched.

## Syllabus navigation

| No. | Practical | Algorithm / Concept | Dataset | Launch |
|:---:|-----------|--------------------|---------|:------:|
| 01 | [Python Basics for ML](01_Python_ML_Basics/) | NumPy, Pandas, Matplotlib | California housing (400 × 10) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/01_Python_ML_Basics/practical_01.ipynb) |
| 02 | [Covariance & Correlation](02_Covariance_Correlation/) | Covariance matrix, Pearson *r* | Iris (150 × 5) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/02_Covariance_Correlation/practical_02.ipynb) |
| 03 | [Linear Regression](03_Linear_Regression/) | OLS, residuals, MSE / RMSE / R² | Medical insurance (1338 × 7) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/03_Linear_Regression/practical_03.ipynb) |
| 04 | [Distance Measures](04_Distance_Measures/) | Euclidean, Manhattan, Cosine | Wine (178 × 14) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/04_Distance_Measures/practical_04.ipynb) |
| 05 | [Decision Tree (ID3)](05_Decision_Tree_ID3/) | Entropy, information gain | Play Tennis (14 × 5) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/05_Decision_Tree_ID3/practical_05.ipynb) |
| 06 | [K-Nearest Neighbours](06_KNN_Classifier/) | Lazy learning, scaling, choosing *k* | Wine (178 × 14) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/06_KNN_Classifier/practical_06.ipynb) |
| 07 | [PCA Feature Reduction](07_PCA_Feature_Reduction/) | Eigen-decomposition, explained variance | Breast cancer (569 × 31) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/07_PCA_Feature_Reduction/practical_07.ipynb) |
| 08 | [Naive Bayes for Text](08_Naive_Bayes_Text/) | Bayes theorem, bag of words | SMS spam (5572 × 2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/08_Naive_Bayes_Text/practical_08.ipynb) |
| 09 | [SVM + Tuning](09_SVM_Hyperparameter_Tuning/) | Max margin, kernels, GridSearchCV | Heart disease (303 × 14) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/09_SVM_Hyperparameter_Tuning/practical_09.ipynb) |
| 10 | [Perceptron Algorithm](10_Perceptron_Algorithm/) | Step activation, delta rule, epochs | Sonar (208 × 61) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamdev07/machine-learning-laboratory/blob/main/10_Perceptron_Algorithm/practical_10.ipynb) |

Lecture and demo notebooks are kept separately in [`theory_class_notebooks/`](theory_class_notebooks/).

## Quick start

### Option 1 — Google Colab (zero setup)

Click any **Open in Colab** badge above, then choose **Runtime → Run all**. Nothing to install and nothing to upload: each notebook reads its CSV straight from this repository.

### Option 2 — Local Jupyter

```bash
git clone https://github.com/liamdev07/machine-learning-laboratory.git
cd machine-learning-laboratory

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Open any `practical_XX.ipynb` and pick the ordinary **Python 3** kernel. Only four libraries are needed: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

### How the data loads

Every notebook chooses its source in one readable line, so the same file works on a lab PC and on Colab:

```python
path = CSV if os.path.exists(CSV) else URL   # local file first, GitHub copy as fallback
df = pd.read_csv(path)
```

## Repository structure

```text
machine-learning-laboratory/
├── 01_Python_ML_Basics/
│   ├── practical_01.ipynb      # the 4-section lab notebook
│   ├── data/                   # CSV used by that notebook
│   └── README.md               # aim, dataset, results, viva Q&A
├── 02_Covariance_Correlation/
│   ...
├── 10_Perceptron_Algorithm/
├── theory_class_notebooks/     # lecture / demo notebooks (not graded labs)
├── requirements.txt
├── LICENSE
└── README.md
```

## 🤝 Open for Open-Source Contributions

This laboratory is **open to batchmates, juniors and any developer** who wants to make it better. You do not need to be an expert — fixing one typo is a perfectly good first pull request.

**Ideas that are always welcome**

- Clearer wording in a *Line-by-Line Code Explanation* section
- Extra viva questions and model answers
- A better plot, a cleaner figure label or an accessibility fix
- New practicals (clustering, regularised regression, neural networks, time series)
- Bug reports where a notebook fails on a different Python or library version

**How to contribute**

1. **Fork** this repository and create a branch: `git checkout -b feature/short-description`
2. Make your change. Keep the **4-section format** and keep the code beginner-readable.
3. **Run the notebook top to bottom** so the saved outputs match the code.
4. Commit with a clear message: `git commit -m "docs: clarify entropy explanation in practical 05"`
5. Push and open a **Pull Request** describing what changed and why.

**Ground rules**

- One practical per pull request keeps reviews quick.
- Do not commit virtual environments, checkpoints or IDE folders — `.gitignore` already covers them.
- Please do not modify `extra/`; it is personal practice work.
- Found a mistake but short on time? Open an **Issue** — that helps just as much.

## Academic use

Treat these notebooks as **lab records**: run them from top to bottom and keep the outputs for submission. Every number quoted in a README is produced by the code in that same notebook, so anything can be reproduced during a viva. Cite the datasets and libraries, and follow your university's academic integrity policy when writing your own record.

## License

Released under the [MIT License](LICENSE) — free to use, study, modify and share with attribution.
