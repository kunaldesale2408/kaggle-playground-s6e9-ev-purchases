# Kaggle Playground Series S6E9: Predicting Electric Vehicle Purchases 🚗⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the end-to-end machine learning solution and experimentation pipeline for the **Kaggle Playground Series (Season 6, Episode 9)** competition: *Predicting Electric Vehicle Purchases*.

---

## 📌 Problem Overview

* **Goal:** Predict whether a consumer will purchase an Electric Vehicle (`Will_Buy_EV`: Yes/No).
* **Dataset Size:** ~530,000 training instances, ~350,000 test instances.
* **Evaluation Metric:** **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve).
* **Validation Strategy:** 5-Fold Stratified K-Fold Cross-Validation (`random_state=42`).

---

## 🏆 Key Results

| Metric | Champion Model (XGBoost 5-Fold) |
|---|---|
| **5-Fold OOF ROC-AUC** | **0.942101** |
| **5-Fold OOF LogLoss** | **0.226011** |
| **Kaggle Public Leaderboard** | **0.941840** |

Fold-by-fold validation performance:
* **Fold 1:** AUC: `0.940981` | LogLoss: `0.227996` (Best Iter: 395)
* **Fold 2:** AUC: `0.941747` | LogLoss: `0.226648` (Best Iter: 439)
* **Fold 3:** AUC: `0.943186` | LogLoss: `0.224161` (Best Iter: 384)
* **Fold 4:** AUC: `0.942630` | LogLoss: `0.225074` (Best Iter: 349)
* **Fold 5:** AUC: `0.941999` | LogLoss: `0.226173` (Best Iter: 407)

---

## 🔍 Data Insights & Feature Engineering

### 1. Synthetic Data Independence
Analysis revealed that numerical features in this synthetic dataset had near-zero linear correlation with each other (e.g. `Age` vs `Annual_Income_USD` correlation = `-0.0045`), consistent with independent marginal distributions.

### 2. High Signal Features (Mutual Information)
Mutual Information (MI) scoring highlighted three dominant signals:
1. `Environmental_Concern_Level` (MI: ~0.132)
2. `Subsidy_Available` (MI: ~0.083)
3. `Annual_Income_USD` (MI: ~0.051)

Features such as `Charging_Stations_Near_Home` and `Age` exhibited marginal direct signal.

### 3. Key Interaction: `Subsidy_x_Income`
The interaction between government subsidies and consumer income proved to be the single most potent leakage-free feature:
```python
subsidy_num = df['Subsidy_Available'].map({'No': 0, 'Yes': 1})
df['Subsidy_x_Income'] = subsidy_num * df['Annual_Income_USD']
```

---

## 🛠 Model Architecture & Hyperparameters

The champion architecture is an **XGBoost Classifier** with native categorical feature support:
* `enable_categorical = True`
* `tree_method = 'hist'`
* `max_depth = 5`
* `learning_rate = 0.10`
* `reg_lambda = 10`
* `min_child_weight = 1`
* `n_estimators = 395` (median iteration across 5 folds with early stopping)

### Why Single XGBoost over Complex Ensembles?
Extensive experiments with:
* Multi-model blends (XGB + LightGBM + CatBoost + HistGradientBoosting)
* Stacking ensembles
* Residual-guided threshold submodels
* Deep learning (PyTorch tabular networks)
* K-Nearest Neighbors pseudo-labeling

yielded only marginal OOF gains (e.g., `+0.00004` to `+0.00008`) while significantly increasing complexity and variance risk. The regularized 5-fold XGBoost model remained the superior choice for production stability and test set generalization.

---

## 📂 Project Structure

```text
├── final_train.py               # Complete training & inference pipeline
├── final_train_out.txt           # Console training execution log
├── FINAL_RESULTS.md             # Summary of findings and competition conclusions
├── data_relationships_report.md # Mutual information & feature correlation analysis
├── notebook.ipynb               # Interactive Jupyter notebook
├── requirements.txt             # Environment dependencies
├── exp3a_results.csv            # Experiment 3a logs
├── exp3b_results.csv            # Experiment 3b logs
├── exp5_blends.csv              # Blending experimentation results
├── exp5_div.csv                 # Model diversity analysis
├── exp5_models.csv              # Individual model baseline comparisons
└── exp7_summary.txt             # Champion model numerical summary
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/kunaldesale2408/kaggle-s6e9-ev-prediction.git
cd kaggle-s6e9-ev-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Place the competition data files (`train.csv`, `test.csv`, `sample_submission.csv`) directly into the repository root:
* Download from Kaggle: [Playground Series S6E9](https://www.kaggle.com/competitions)

### 4. Run Training & Inference
```bash
python final_train.py
```
This will:
1. Preprocess data and construct `Subsidy_x_Income`.
2. Execute 5-fold Stratified K-Fold cross-validation.
3. Fit the full dataset using median optimal iterations (`n_estimators=395`).
4. Generate `submission.csv` with automated data integrity assertions.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
