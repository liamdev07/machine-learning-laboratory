# Practical 09 — Support Vector Machines and Hyperparameter Tuning

**Syllabus (Parul University):** *Implement a Support Vector Machine and tune hyperparameters.*  
**Kernel:** `ml-lab-kernel`  
**Data:** UCI-style heart disease table (`data/heart_disease.csv`) — binary `target`

## Objectives

1. State the **maximum-margin** primal, **Lagrangian dual**, slack **\(C\)**, and **linear / poly / RBF** kernels (\(\gamma\)).
2. Reconstruct sklearn’s **dual decision function** \(f(x)=\sum_i \alpha_i y_i K(x_i,x)+b\) in NumPy.
3. Fit `Pipeline([StandardScaler, SVC])` with a **train-only** scaler.
4. `GridSearchCV` over `C`, `gamma`, and `kernel`.
5. Plot 2-D regions (PCA plane) with **margin contours** \(\{f=\pm 1\}\) and **circled support vectors**; heatmap of CV accuracy vs \((C,\gamma)\) for RBF.
6. Accuracy, confusion matrix, ROC-AUC.

## Dual load path

1. `./data/heart_disease.csv`
2. Public GitHub CSV (same schema)
3. `sklearn.datasets.load_breast_cancer` if the network path fails

## How to run

```bash
cd 09_SVM_Hyperparameter_Tuning
jupyter notebook practical_09.ipynb
```
