# FINAL_RESULTS

1. **Competition name:** Kaggle Playground Series S6E9 - Predicting Electric Vehicle Purchases
2. **Dataset dimensions:** ~530,000 rows (train), ~350,000 rows (test)
3. **Target:** `Will_Buy_EV` (Binary classification: Yes/No)
4. **Evaluation metric:** ROC-AUC
5. **Final model:** XGBoost Classifier (native categorical handling, 5-fold StratifiedKFold)
6. **Final features:** All original categorical and numerical features (excluding `id`) + one engineered feature: `Subsidy_x_Income`
7. **Hyperparameters:** `max_depth = 5`, `learning_rate = 0.10`, `reg_lambda = 10`, `min_child_weight = 1`, `tree_method = 'hist'`, `n_estimators ≈ 395` (early stopping)
8. **5-fold OOF AUC:** 0.942101
9. **OOF LogLoss:** 0.226011
10. **Kaggle public LB score:** 0.941840
11. **Best alternative tested:** Generic Rank Averaging Ensemble (XGB + LGBM + CatBoost + HGBC) which reached ≈0.9422, and Residual-guided Threshold Submodels which reached ≈0.942141.
12. **Why the alternatives were rejected:** The maximum observed improvements over the baseline were highly marginal (e.g., ~+0.000040 to +0.000080), failing to cross the pre-defined +0.0005 "Credible" improvement threshold. They were not meaningful enough to justify replacing the structurally sound, heavily regularized champion model.
13. **Final submission filename:** `submission.csv`
14. **Exact conclusion:** After 14 experiments covering model comparison, hyperparameter optimization, feature engineering, ensembles, residual analysis, nonlinear transformations, gated models, threshold discovery, latent-score reconstruction, and synthetic-target reverse engineering, no meaningful leakage-free improvement over the 0.942101 OOF champion was demonstrated. The exact mechanism behind the public 0.946+ leaderboard scores remains unexplained because the corresponding competitor implementations and prediction artifacts were not recoverable. The existing 0.941840 Kaggle submission therefore remains the selected final submission.
