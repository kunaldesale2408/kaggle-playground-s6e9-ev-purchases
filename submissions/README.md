# Submissions Directory

This directory contains the final generated test prediction CSV files submitted to the [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) competition.

## 📊 Summary of Submissions

| File Name | Experiment Description | CV / OOF AUC | LB Score | Notes |
|---|---|---|---|---|
| **`submission.csv`** | **Champion Model** (5-Fold Stratified XGBoost with native categorical support and `Subsidy_x_Income` interaction) | **0.942101** | **0.941840** | **Final Selected Submission** — Structurally robust, regularized champion model. |
| `submission_exp15_stack.csv` | Multi-model Stacking (XGBoost + LightGBM + CatBoost meta-learner) | 0.942180 | 0.941850 | Marginal gain over single model (+0.00008), higher complexity. |
| `submission_exp16_pseudo.csv` | Test Pseudo-labeling (High-confidence pseudo-labels injected into train) | 0.942080 | 0.941790 | Slight degradation due to synthetic noise reinforcement. |
| `submission_exp17_pytorch.csv` | PyTorch Deep Tabular Network with embedding layers for categoricals | 0.938520 | 0.938100 | Solid neural baseline, but tree-based models outperformed tabular deep learning. |
| `submission_exp18_grand_ensemble.csv` | Grand Rank-Averaged Ensemble (Tree models + PyTorch Tabular) | 0.942210 | 0.941860 | Top ensemble score, but did not reach the pre-defined +0.0005 credibility bar. |
| `submission_exp19_knn.csv` | KNN Feature-Assisted Hybrid Model | 0.941950 | 0.941720 | Distance metrics added minimal value on uncorrelated synthetic features. |

## ℹ️ Target Format
* `id`: Test row identifier
* `Will_Buy_EV`: Predicted purchase probability (continuous float in range `[0.0, 1.0]`)
