import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent

def resolve_path(filename):
    p = BASE_DIR / filename
    if p.exists():
        return p
    alt = Path("D:/Kaggle/Competition1") / filename
    return alt if alt.exists() else p

print("=== STEP 1: Loading Data & Preprocessing ===")
train = pd.read_csv(resolve_path("train.csv"))
test = pd.read_csv(resolve_path("test.csv"))
sample_sub = pd.read_csv(resolve_path("sample_submission.csv"))

y = (train['Will_Buy_EV'] == 'Yes').astype(int)
train_id = train['id']
test_id = test['id']

train = train.drop(['Will_Buy_EV', 'id'], axis=1)
test = test.drop(['id'], axis=1)

def preprocess(df):
    df_mod = df.copy()
    subsidy_num = df_mod['Subsidy_Available'].map({'No': 0, 'Yes': 1})
    df_mod['Subsidy_x_Income'] = subsidy_num * df_mod['Annual_Income_USD']
    cat_cols = df_mod.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        df_mod[c] = df_mod[c].astype('category')
    return df_mod

X = preprocess(train)
X_test = preprocess(test)

print("=== STEP 2: 5-Fold OOF Evaluation ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
fold_aucs = []
fold_lls = []
best_iters = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[train_idx]
    X_va = X.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_va = y.iloc[val_idx]
    
    model = XGBClassifier(
        random_state=42, enable_categorical=True, tree_method='hist', n_jobs=-1,
        max_depth=5, learning_rate=0.10, reg_lambda=10, min_child_weight=1, n_estimators=1500,
        early_stopping_rounds=50
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    best_iter = model.best_iteration
    best_iters.append(best_iter)
    
    preds = model.predict_proba(X_va, iteration_range=(0, best_iter + 1))[:, 1]
    oof_preds[val_idx] = preds
    
    auc = roc_auc_score(y_va, preds)
    ll = log_loss(y_va, preds)
    fold_aucs.append(auc)
    fold_lls.append(ll)
    
    print(f"Fold {fold+1} | AUC: {auc:.6f} | LogLoss: {ll:.6f} | Best Iter: {best_iter}")

overall_auc = roc_auc_score(y, oof_preds)
overall_ll = log_loss(y, oof_preds)
print(f"\nFinal OOF AUC: {overall_auc:.6f}")
print(f"Final OOF LogLoss: {overall_ll:.6f}")

print("\n=== STEP 3: Final Tree Count Selection ===")
median_iter = int(np.median(best_iters))
print(f"Fold Iters: {best_iters}")
print(f"Mean Iter: {np.mean(best_iters):.1f}")
print(f"Median Iter: {median_iter}")
print(f"Std Iter: {np.std(best_iters):.1f}")
print(f"Min Iter: {np.min(best_iters)}, Max Iter: {np.max(best_iters)}")

print("\n=== STEP 4: Training Full Model ===")
final_model = XGBClassifier(
    random_state=42, enable_categorical=True, tree_method='hist', n_jobs=-1,
    max_depth=5, learning_rate=0.10, reg_lambda=10, min_child_weight=1, n_estimators=median_iter
)
# No early stopping
final_model.fit(X, y)

print("\n=== STEP 5: Generating Test Predictions ===")
test_preds = final_model.predict_proba(X_test)[:, 1]

print("Test Prediction Stats:")
print(f"Min: {test_preds.min():.6f}, Max: {test_preds.max():.6f}")
print(f"Mean: {test_preds.mean():.6f}, Std: {test_preds.std():.6f}")
print(f"Unique preds: {len(np.unique(test_preds))}")
print(f"NaNs: {np.isnan(test_preds).sum()}, Infs: {np.isinf(test_preds).sum()}")

print("\n=== STEP 6: Formatting and Integrity Checks ===")
id_col = sample_sub.columns[0]
target_col = sample_sub.columns[1]

submission = pd.DataFrame({
    id_col: test_id,
    target_col: test_preds
})

assert len(submission) == len(test), "Length mismatch with test data!"
assert len(submission) == len(sample_sub), "Length mismatch with sample_submission!"
assert list(submission.columns) == list(sample_sub.columns), "Column names do not match sample_submission!"
assert submission[id_col].equals(sample_sub[id_col]), "ID column does not match sample_submission order/values!"
assert submission[target_col].isna().sum() == 0, "NaNs found in submission target!"
assert np.isinf(submission[target_col]).sum() == 0, "Infs found in submission target!"
assert submission[target_col].min() >= 0.0 and submission[target_col].max() <= 1.0, "Probabilities out of bounds!"

print("All integrity checks passed successfully!")

sub_path = BASE_DIR / "submission.csv"
submission.to_csv(sub_path, index=False)
print(f"Saved final submission to {sub_path}")

# Save summary to file for artifact generation later
summary_path = BASE_DIR / "exp7_summary.txt"
with open(summary_path, "w") as f:
    f.write(f"{overall_auc},{overall_ll},{best_iters},{median_iter},{test_preds.min()},{test_preds.max()},{test_preds.mean()},{test_preds.std()},{len(np.unique(test_preds))}")
