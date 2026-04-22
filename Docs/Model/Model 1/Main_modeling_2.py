import warnings
import numpy as np
import pandas as pd


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


def evaluate_regression(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


# This file expects Main_Preprocessing_1.py to be run first in the notebook.
# It uses df6 that should already exist in memory.

''' 
target_col = "LN_IC50"

candidate_feature_cols = [
    "DRUG_NAME",
    "TARGET_PATHWAY",
    "TCGA_DESC",
    "Microsatellite instability Status (MSI)",
    "Growth Properties",
    "GDSC Tissue descriptor 1",
    "GDSC Tissue descriptor 2",
    "mutational_burden",
    "ploidy_snp6",
    "ploidy_wes",
]
candidate_feature_cols = [c for c in candidate_feature_cols if c in df6.columns]

X_candidate = df6[candidate_feature_cols].copy()
y = df6[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_candidate,
    y,
    test_size=0.20,
    random_state=42,
)

X_train = X_train.copy()
X_test = X_test.copy()

if "DRUG_NAME" in X_train.columns:
    X_train["DRUG_NAME"] = X_train["DRUG_NAME"].astype("string").fillna("MISSING")
    X_test["DRUG_NAME"] = X_test["DRUG_NAME"].astype("string").fillna("MISSING")

    drug_mean_map = y_train.groupby(X_train["DRUG_NAME"]).mean()
    global_train_mean = float(y_train.mean())

    X_train["DRUG_NAME_target_enc"] = (
        X_train["DRUG_NAME"].map(drug_mean_map).fillna(global_train_mean)
    )
    X_test["DRUG_NAME_target_enc"] = (
        X_test["DRUG_NAME"].map(drug_mean_map).fillna(global_train_mean)
    )

    X_train = X_train.drop(columns=["DRUG_NAME"])
    X_test = X_test.drop(columns=["DRUG_NAME"])

ohe_cols = [
    "TCGA_DESC",
    "GDSC Tissue descriptor 1",
    "GDSC Tissue descriptor 2",
    "Microsatellite instability Status (MSI)",
    "Growth Properties",
    "TARGET_PATHWAY",
]
ohe_cols = [c for c in ohe_cols if c in X_train.columns]

numeric_cols = [c for c in X_train.columns if c not in ohe_cols]

for col in ohe_cols:
    X_train[col] = X_train[col].astype("string").fillna("MISSING")
    X_test[col] = X_test[col].astype("string").fillna("MISSING")

for col in numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

train_medians = X_train[numeric_cols].median()
X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
X_test[numeric_cols] = X_test[numeric_cols].fillna(train_medians)

X_train_cat = pd.get_dummies(X_train[ohe_cols], columns=ohe_cols, dummy_na=False)
X_test_cat = pd.get_dummies(X_test[ohe_cols], columns=ohe_cols, dummy_na=False)
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join="left", axis=1, fill_value=0)

X_train_encoded = pd.concat([X_train_cat, X_train[numeric_cols]], axis=1)
X_test_encoded = pd.concat([X_test_cat, X_test[numeric_cols]], axis=1)

feature_map = pd.DataFrame(
    {
        "original_name": X_train_encoded.columns,
        "safe_name": [f"f_{i:04d}" for i in range(X_train_encoded.shape[1])],
    }
)

X_train_model = X_train_encoded.copy()
X_test_model = X_test_encoded.copy()
X_train_model.columns = feature_map["safe_name"].tolist()
X_test_model.columns = feature_map["safe_name"].tolist()

print("Modeling data used:")
print(f"  X_train rows: {X_train_model.shape[0]}")
print(f"  X_test rows : {X_test_model.shape[0]}")
print(f"  Features    : {X_train_model.shape[1]}")

'''

# This file expects Main_Preprocessing_2.py to be run first in the notebook.
# It uses X_train_scaled, X_test_scaled, y_train, and y_test already created in memory.

required_vars = ["X_train_scaled", "X_test_scaled", "y_train", "y_test"]
missing_vars = [name for name in required_vars if name not in globals()]
if missing_vars:
    raise NameError(
        'Run %run "Main_Preprocessing_2.py" in the previous notebook cell first. '
        "Missing variables: " + ", ".join(missing_vars)
    )

# Use preprocessed matrices directly
X_train_model = X_train_scaled.copy()
X_test_model = X_test_scaled.copy()

# Safe feature names help XGBoost and LightGBM handle special characters
feature_map = pd.DataFrame(
    {
        "original_name": X_train_model.columns,
        "safe_name": [f"f_{i:04d}" for i in range(X_train_model.shape[1])],
    }
)

X_train_safe = X_train_model.copy()
X_test_safe = X_test_model.copy()
X_train_safe.columns = feature_map["safe_name"].tolist()
X_test_safe.columns = feature_map["safe_name"].tolist()

print("Modeling data used:")
print(f"  X_train rows: {X_train_model.shape[0]}")
print(f"  X_test rows : {X_test_model.shape[0]}")
print(f"  Features    : {X_train_model.shape[1]}")



# Model parameters that we got from optuna
catboost_params = {
    "iterations": 1500,
    "learning_rate": 0.13558330321686268,
    "depth": 10,
    "l2_leaf_reg": 0.01578981814610257,
    "subsample": 0.7502720766157313,
    "random_strength": 0.0035710492569881653,
    "bagging_temperature": 2.863575984094419,
    "min_data_in_leaf": 12,
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "verbose": 0,
    "random_state": 42,
}

xgboost_params = {
    "n_estimators": 1600,
    "learning_rate": 0.0606212053346252,
    "max_depth": 10,
    "min_child_weight": 3,
    "subsample": 0.8664841675523584,
    "colsample_bytree": 0.663669197027016,
    "gamma": 0.41282236439827835,
    "reg_alpha": 0.001139884908293192,
    "reg_lambda": 5.4367797220761656,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

lightgbm_params = {
    "max_depth": 13,
    "n_estimators": 1750,
    "learning_rate": 0.05055647998564387,
    "num_leaves": 183,
    "min_child_samples": 10,
    "subsample": 0.8466836046614781,
    "colsample_bytree": 0.6984490363901545,
    "reg_alpha": 0.26189360447192045,
    "reg_lambda": 0.325726294788637,
    "objective": "regression",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


print("\nCatBoost tuned")
print("Hyperparameters:")
for k, v in catboost_params.items():
    print(f"  {k}: {v}")

cat_model = CatBoostRegressor(**catboost_params)
cat_model.fit(X_train_model, y_train, verbose=False)
y_pred_cat = cat_model.predict(X_test_model)
cat_rmse, cat_mae, cat_r2 = evaluate_regression(y_test, y_pred_cat)

print("Metrics:")
print(f"  RMSE: {cat_rmse:.6f}")
print(f"  MAE : {cat_mae:.6f}")
print(f"  R2  : {cat_r2:.6f}")


print("\nXGBoost tuned")
print("Hyperparameters:")
for k, v in xgboost_params.items():
    print(f"  {k}: {v}")

xgb_model = XGBRegressor(**xgboost_params)
xgb_model.fit(X_train_safe, y_train)
y_pred_xgb = xgb_model.predict(X_test_safe)
xgb_rmse, xgb_mae, xgb_r2 = evaluate_regression(y_test, y_pred_xgb)

print("Metrics:")
print(f"  RMSE: {xgb_rmse:.6f}")
print(f"  MAE : {xgb_mae:.6f}")
print(f"  R2  : {xgb_r2:.6f}")


print("\nLightGBM tuned")
print("Hyperparameters:")
for k, v in lightgbm_params.items():
    print(f"  {k}: {v}")

lgbm_model = LGBMRegressor(**lightgbm_params)
lgbm_model.fit(X_train_safe, y_train)
y_pred_lgbm = lgbm_model.predict(X_test_safe)
lgb_rmse, lgb_mae, lgb_r2 = evaluate_regression(y_test, y_pred_lgbm)

print("Metrics:")
print(f"  RMSE: {lgb_rmse:.6f}")
print(f"  MAE : {lgb_mae:.6f}")
print(f"  R2  : {lgb_r2:.6f}")


results_df = pd.DataFrame(
    [
        {"Model": "XGBoost tuned", "RMSE": xgb_rmse, "MAE": xgb_mae, "R2": xgb_r2},
        {"Model": "CatBoost tuned", "RMSE": cat_rmse, "MAE": cat_mae, "R2": cat_r2},
        {"Model": "LightGBM tuned", "RMSE": lgb_rmse, "MAE": lgb_mae, "R2": lgb_r2},
    ]
).sort_values("R2", ascending=False).reset_index(drop=True)

print("\nRanked model comparison:")
print(results_df.to_string(index=False))