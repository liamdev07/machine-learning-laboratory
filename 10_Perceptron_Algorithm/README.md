# Practical 10 — Perceptron Learning Algorithm

**Syllabus (Parul University):** *Implement the Perceptron learning algorithm and evaluate performance.*  
**Kernel:** `ml-lab-kernel`  
**Data:** Connectionist Bench Sonar — Mines vs Rocks (`data/sonar.csv`, 208 × 60 + label)

## Objectives

1. State the neuron model, Heaviside activation, Rosenblatt **delta** update, and **Novikoff** mistake bound.
2. Implement NumPy `Perceptron` with **misclassifications per epoch**.
3. Show **AND/OR converge**, **XOR does not** (Minsky & Papert).
4. `sklearn.linear_model.Perceptron` + `StandardScaler` on Sonar.
5. Error-vs-epoch, 2-D boundary snapshots, AND vs XOR geometry.
6. Confusion matrix, precision, recall.

## Dual load path

1. `./data/sonar.csv`
2. GitHub / UCI sonar CSV (no header)
3. Embedded fallback is not a substitute for the 208-row bench

## How to run

```bash
cd 10_Perceptron_Algorithm
jupyter notebook practical_10.ipynb
```
