# Kaggle Playground Series S6E9: Predicting Electric Vehicle Purchases 🚗⚡

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E9-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e9)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Kaggle Competition:** [Playground Series - Season 6, Episode 9: Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9)

This repository contains the complete machine learning pipeline, exploratory data profiling, feature engineering experiments, and final model submission artifacts for the Kaggle Playground Series S6E9 competition.

---

## 📌 Problem Description & Objectives

* **Task:** Binary classification to predict consumer willingness to purchase an Electric Vehicle (`Will_Buy_EV`: `Yes` / `No`).
* **Evaluation Metric:** **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve).
* **Dataset Scale:**
  * **Train Set:** ~530,000 observations
  * **Test Set:** ~350,000 observations
* **Validation Strategy:** Leakage-free 5-Fold Stratified K-Fold Cross-Validation (`random_state=42`).

---

## 🏆 Benchmark & Leaderboard Results

| Model / Pipeline | CV / OOF ROC-AUC | OOF LogLoss | Kaggle Public LB | Status |
|---|---|---|---|---|
| **Champion Model (5-Fold XGBoost + Interaction)** | **0.942101** | **0.226011** | **0.941840** | **Selected Final** |
| Grand Rank Ensemble (Tree Models + Neural Net) | 0.942210 | 0.225980 | 0.941860 | Rejected (Marginal gain: +0.00002) |
| Stacking Meta-Learner (XGB + LGBM + CatBoost) | 0.942180 | 0.226000 | 0.941850 | Rejected (High variance complexity) |
| PyTorch Deep Tabular Network | 0.938520 | 0.231200 | 0.938100 | Baseline neural benchmark |
| Test Pseudo-Labeling | 0.942080 | 0.226150 | 0.941790 | Overfitting on synthetic noise |

### 5-Fold Cross-Validation Breakdown (Champion Model)
* **Fold 1:** ROC-AUC: `0.940981` | LogLoss: `0.227996` | Best Iteration: 395
* **Fold 2:** ROC-AUC: `0.941747` | LogLoss: `0.226648` | Best Iteration: 439
* **Fold 3:** ROC-AUC: `0.943186` | LogLoss: `0.224161` | Best Iteration: 384
* **Fold 4:** ROC-AUC: `0.942630` | LogLoss: `0.225074` | Best Iteration: 349
* **Fold 5:** ROC-AUC: `0.941999` | LogLoss: `0.226173` | Best Iteration: 407
* **Overall Out-Of-Fold (OOF):** ROC-AUC: `0.942101` | LogLoss: `0.226011`

---

## 🔍 Key Data Insights & Engineering

### 1. Synthetic Feature Independence
Comprehensive feature correlation profiling revealed that numerical features in this competition dataset were synthetically generated with near-zero mutual linear correlation (e.g. `Age` vs. `Annual_Income_USD` correlation = `-0.0045`).

### 2. Information Dominance
Mutual Information (MI) analysis demonstrated that signal is concentrated in a few key features:
1. `Environmental_Concern_Level` (MI: `0.1316`)
2. `Subsidy_Available` (MI: `0.0825`)
3. `Annual_Income_USD` (MI: `0.0513`)

### 3. High-Leverage Interaction: `Subsidy_x_Income`
The interaction between governmental EV financial subsidies and personal annual income provided a decisive, leakage-free predictive lift:
```python
subsidy_binary = df['Subsidy_Available'].map({'No': 0, 'Yes': 1})
df['Subsidy_x_Income'] = subsidy_binary * df['Annual_Income_USD']
```

---

## 🛠 Champion Model Architecture

The champion solution employs **XGBoost Classifier** with native histogram-based categorical support:
* `enable_categorical = True`
* `tree_method = 'hist'`
* `max_depth = 5`
* `learning_rate = 0.10`
* `reg_lambda = 10`
* `min_child_weight = 1`
* `n_estimators = 395` (Median of fold early-stopping iterations)

---

## 📂 Repository Structure

```text
├── submissions/                                # Directory containing all competition submissions
│   ├── README.md                               # Detailed guide to all submission files & scores
│   ├── submission.csv                          # Champion model test predictions (LB: 0.941840)
│   ├── submission_exp15_stack.csv              # Stacking ensemble predictions
│   ├── submission_exp16_pseudo.csv             # Pseudo-labeled model predictions
│   ├── submission_exp17_pytorch.csv            # PyTorch tabular model predictions
│   ├── submission_exp18_grand_ensemble.csv     # Grand rank-averaged ensemble predictions
│   └── submission_exp19_knn.csv                # KNN feature-assisted predictions
├── final_train.py                              # End-to-end training, CV & inference script
├── final_train_out.txt                          # Terminal training log
├── FINAL_RESULTS.md                            # Comprehensive experiment decision summary
├── data_relationships_report.md                # Data relationship and mutual information report
├── notebook.ipynb                              # Interactive Jupyter notebook
├── requirements.txt                            # Python dependencies
├── exp3a_results.csv                           # Experiment 3a log
├── exp3b_results.csv                           # Experiment 3b log
├── exp5_blends.csv                             # Blending experiment log
├── exp5_div.csv                                # Diversity score metrics
├── exp5_models.csv                             # Model comparisons
├── exp7_summary.txt                            # Final metrics summary
└── LICENSE                                     # MIT License
```

---

## 🚀 How to Reproduce

### 1. Clone this Repository
```bash
git clone https://github.com/kunaldesale2408/kaggle-playground-s6e9-ev-purchases.git
cd kaggle-playground-s6e9-ev-purchases
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download Competition Data
Download data files directly from the [Kaggle S6E9 Competition Page](https://www.kaggle.com/competitions/playground-series-s6e9/data) or using the Kaggle CLI:
```bash
kaggle competitions download -c playground-series-s6e9
unzip playground-series-s6e9.zip -d .
```
Ensure `train.csv`, `test.csv`, and `sample_submission.csv` are in the root directory.

### 4. Run Training & Prediction
```bash
python final_train.py
```
This script will:
* Build the `Subsidy_x_Income` feature.
* Execute 5-fold Stratified K-Fold validation and print fold metrics.
* Fit on the entire training set using the optimal tree count.
* Validate all prediction distributions and format constraints.
* Save the output to `submissions/submission.csv`.

---

## 🔗 Competition Link
* **Official Kaggle Competition:** [https://www.kaggle.com/competitions/playground-series-s6e9](https://www.kaggle.com/competitions/playground-series-s6e9)

---

## 📜 License
Distributed under the [MIT License](LICENSE).
