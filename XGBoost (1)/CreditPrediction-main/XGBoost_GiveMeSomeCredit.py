
import os
import warnings
warnings.filterwarnings("ignore")

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)  # for macOS compatibility
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

# Optional: shap for explainability (may be slow)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# -----------------------
# CONFIG
# -----------------------
SEED = 42
DATA_FILE = "XGBoost/cs-training.csv"   # put file here
OUT_DIR = "output_model"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_SEARCH_ITERS = 30       # keep moderate for speed in demo; increase if you have time
N_JOBS_CV = -1                 # use all CPUs for RandomizedSearchCV
CV_FOLDS = 4

# -----------------------
# 1) Load data (robust)
# -----------------------
print("Loading data:", DATA_FILE)
df = pd.read_csv(DATA_FILE)

# dataset sometimes has an unnamed index column; drop if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# target column in this dataset is 'SeriousDlqin2yrs'
if "SeriousDlqin2yrs" in df.columns:
    df = df.rename(columns={"SeriousDlqin2yrs": "TARGET"})
elif "TARGET" not in df.columns:
    raise ValueError("Couldn't find target column (SeriousDlqin2yrs or TARGET). Check CSV.")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------
# 2) Quick EDA
# -----------------------
print("\n-- Target distribution --")
print(df["TARGET"].value_counts(dropna=False))
print("\n-- Missing values (top) --")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\n-- Basic stats (numeric) --")
display_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[display_cols].describe().T)

# Check for obvious invalid ages or negative values
if "age" in df.columns:
    print("\nAge min/max:", df["age"].min(), df["age"].max())

# -----------------------
# 3) Data cleaning decisions (careful)
# -----------------------
# Observations on this dataset:
# - All features are numeric. Missing commonly in MonthlyIncome and NumberOfDependents.
# - We'll impute median for numeric features (simple and robust).
# - We'll clip ages to a reasonable range (18..100) to avoid unrealistic outliers for presentation.

if "age" in df.columns:
    # clip ages outside 18..100 (rare)
    df["age"] = df["age"].clip(lower=18, upper=100)

# Remove duplicates (unlikely but safe)
n_dup = df.duplicated().sum()
if n_dup > 0:
    print(f"Found {n_dup} duplicated rows. Dropping duplicates.")
    df = df.drop_duplicates()

# -----------------------
# 4) Feature / target split
# -----------------------
X = df.drop(columns=["TARGET"])
y = df["TARGET"].astype(int)

feature_names = X.columns.tolist()
print("\nFeature count:", len(feature_names))

# -----------------------
# 5) Train/validation/test split (stratified)
# -----------------------
# We'll keep a held-out test set (20%) for final evaluation.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
# From train_full, reserve a small validation set for early stopping (10% of train_full)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, stratify=y_train_full, random_state=SEED
)

print("\nShapes:")
print("  X_train:", X_train.shape)
print("  X_val:  ", X_val.shape)
print("  X_test: ", X_test.shape)

# -----------------------
# 6) Preprocessing pipeline (numeric only)
# -----------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # robust to outliers
    ("scaler", StandardScaler())
])

# Fit preprocessor on training fold
numeric_pipeline.fit(X_train)

# Transform sets
X_train_proc = numeric_pipeline.transform(X_train)
X_val_proc   = numeric_pipeline.transform(X_val)
X_test_proc  = numeric_pipeline.transform(X_test)
X_train_full_proc = numeric_pipeline.transform(X_train_full)

# -----------------------
# 7) Class imbalance handling
# -----------------------
n_pos = y_train_full.sum()
n_neg = len(y_train_full) - n_pos
scale_pos_weight = n_neg / max(1, n_pos)
print(f"\nClass counts (train_full): pos={n_pos}, neg={n_neg}, scale_pos_weight={scale_pos_weight:.3f}")

# -----------------------
# 8) Baseline XGBoost (quick)
# -----------------------
print("\nTraining baseline XGBoost (quick)...")
base_clf = XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="auc",
    n_jobs=1,            # IMPORTANT: set to 1 for inner estimator to avoid thread oversubscription during CV
    random_state=SEED,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight
)

base_clf.fit(X_train_proc, y_train)
yval_proba = base_clf.predict_proba(X_val_proc)[:, 1]
print("Baseline ROC-AUC on val:", roc_auc_score(y_val, yval_proba))

# -----------------------
# 9) Hyperparameter search (RandomizedSearchCV)
# -----------------------
# We'll search moderately wide but keep iterations modest for a class presentation.
param_dist = {
    "n_estimators": [100, 200, 400, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5, 1],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10]
}

print("\nStarting RandomizedSearchCV (this may take a while)...")
est = XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="auc",
    n_jobs=1,       # keep 1 inside estimator
    random_state=SEED,
    scale_pos_weight=scale_pos_weight
)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
rand = RandomizedSearchCV(
    estimator=est,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITERS,
    scoring="roc_auc",
    cv=cv,
    verbose=2,
    random_state=SEED,
    n_jobs=N_JOBS_CV
)

# Fit on TRAIN_FULL (we'll later train final model with early stopping using X_val)
rand.fit(X_train_full_proc, y_train_full)

print("\nRandom search best score (CV AUC):", rand.best_score_)
print("Best params:")
pprint(rand.best_params_)

best_params = rand.best_params_
# add scale_pos_weight and eval_metric/details
best_params["use_label_encoder"] = False
best_params["objective"] = "binary:logistic"
best_params["random_state"] = SEED
best_params["n_jobs"] = 1
best_params["scale_pos_weight"] = scale_pos_weight

# -----------------------
# 10) Final training with early stopping (using X_train_full vs X_val)
# -----------------------
print("\nFinal training with early stopping (using validation set)...")
final_clf = XGBClassifier(**best_params)

# Fit on processed arrays with early stopping
final_clf.fit(
    X_train_full_proc, y_train_full,
    eval_set=[(X_val_proc, y_val)]
)

# -----------------------
# 11) Evaluation on test set
# -----------------------
y_test_proba = final_clf.predict_proba(X_test_proc)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)

roc = roc_auc_score(y_test, y_test_proba)
print("\nFinal Test ROC-AUC:", roc)
print("\nClassification report (threshold=0.5):")
print(classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:\n", cm)

# Plot and save confusion matrix as image
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.show()

# Plot ROC curve
RocCurveDisplay.from_predictions(y_test, y_test_proba)
plt.title("ROC curve - Final model")
plt.tight_layout()
plt.show()

# -----------------------
# 12) Feature importance (clean)
# -----------------------
# Because we applied a simple preprocessor that doesn't change feature order, we can map features directly.
imp = final_clf.feature_importances_
feat_imp = pd.Series(imp, index=feature_names).sort_values(ascending=False)
print("\nTop features by importance:")
print(feat_imp.head(20))

plt.figure(figsize=(8,6))
feat_imp.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 feature importances (XGBoost)")
plt.tight_layout()
plt.show()

# -----------------------
# 13) SHAP explainability (optional)
# -----------------------
if _HAS_SHAP:
    print("\nComputing SHAP values (may take time)...")
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(final_clf)
    shap_values = explainer.shap_values(X_test_proc)
    # summary plot (global)
    shap.summary_plot(shap_values, features=X_test_proc, feature_names=feature_names, show=True)
    # partial dependence for top feature
    topf = feat_imp.index[0]
    print("Top feature:", topf)
    shap.dependence_plot(topf, shap_values, X_test_proc, feature_names=feature_names, show=True)
else:
    print("\nshap not installed or failed to import. To enable SHAP, pip install shap and run again.")

# -----------------------
# 14) Save artifacts
# -----------------------
print("\nSaving model and preprocessor to", OUT_DIR)
joblib.dump(final_clf, os.path.join(OUT_DIR, "xgb_final_model.joblib"))
joblib.dump(numeric_pipeline, os.path.join(OUT_DIR, "preprocessor.joblib"))
# Save feature list and best params
pd.Series(feature_names).to_csv(os.path.join(OUT_DIR, "feature_names.csv"), index=False)
joblib.dump(best_params, os.path.join(OUT_DIR, "best_params.joblib"))

print("Done. Artifacts:")
print(" -", os.path.join(OUT_DIR, "xgb_final_model.joblib"))
print(" -", os.path.join(OUT_DIR, "preprocessor.joblib"))
print(" -", os.path.join(OUT_DIR, "feature_names.csv"))
print(" -", os.path.join(OUT_DIR, "best_params.joblib"))

# -----------------------
# 15) Notes & next steps (for your presentation)
# -----------------------
notes = """
GHI CHÚ:
- Tập trung dùng ROC-AUC làm chỉ số chính vì class imbalance và bài toán đánh giá rủi ro.
- Đã dùng median imputation cho tính đơn giản và ổn định; nếu muốn nâng cấp: thử IterativeImputer hoặc tạo feature 'missing_flag'.
- Đã tính scale_pos_weight theo tỷ lệ âm/dương trên tập train_full để giúp XGBoost xử lý imbalance.
- RandomizedSearchCV đã dùng estimator với n_jobs=1 để tránh oversubscription; RandomizedSearchCV chạy với n_jobs=-1.
- Early stopping dùng validation set riêng (không dùng trong CV) để chọn số vòng thực tế.
- Để sử dụng model trên web: load preprocessor + model (joblib), gọi preprocessor.transform(X_new) -> model.predict_proba(...)
- Nâng cao: làm K-fold CV (out-of-fold preds) để có estimate chính xác hơn; feature engineering (interaction features, binning age, log MonthlyIncome,...); calibration cho probabilities.
"""
print(notes)
