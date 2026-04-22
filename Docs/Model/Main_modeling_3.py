import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# This function calculates the main regression metrics.
def evaluate_regression(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


# This checks that preprocessing was already run in the notebook.
required_vars = ["X_train_scaled", "X_test_scaled", "y_train", "y_test"]
missing_vars = [name for name in required_vars if name not in globals()]
if missing_vars:
    raise NameError(
        'Run %run "Main_Preprocessing_2.py" in the previous notebook cell first. '
        "Missing variables: " + ", ".join(missing_vars)         )

# This uses the preprocessed train and test matrices directly.
X_train_model = X_train_scaled.copy()
X_test_model = X_test_scaled.copy()

# This creates safe feature names for XGBoost.
feature_map = pd.DataFrame(
    {        "original_name": X_train_model.columns,
        "safe_name": [f"f_{i:04d}" for i in range(X_train_model.shape[1])],     }        )

X_train_safe = X_train_model.copy()
X_test_safe = X_test_model.copy()
X_train_safe.columns = feature_map["safe_name"].tolist()
X_test_safe.columns = feature_map["safe_name"].tolist()

print("Modeling data used:")
print(f"  X_train rows: {X_train_model.shape[0]}")
print(f"  X_test rows : {X_test_model.shape[0]}")
print(f"  Features    : {X_train_model.shape[1]}")

# This creates a validation split for the stacking layer.
X_base_train_cat, X_meta_valid_cat, y_base_train, y_meta_valid = train_test_split(
    X_train_model,
    y_train,
    test_size=0.20,
    random_state=42 )

X_base_train_xgb = X_base_train_cat.copy()
X_meta_valid_xgb = X_meta_valid_cat.copy()
X_base_train_xgb.columns = feature_map["safe_name"].tolist()
X_meta_valid_xgb.columns = feature_map["safe_name"].tolist()

print("\nStacking split:")
print(f"  Base train rows: {X_base_train_cat.shape[0]}")
print(f"  Meta valid rows: {X_meta_valid_cat.shape[0]}")

# These are the hyper tuned parameters for CatBoost.
catboost_params = {
    "iterations": 2100,
    "learning_rate": 0.12722539009397227,
    "depth": 12,
    "l2_leaf_reg": 4.1848395732303,
    "subsample": 0.8160378134726768,
    "random_strength": 0.22997799412225067,
    "bagging_temperature": 0.4303201189380161,
    "min_data_in_leaf": 22,
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "verbose": 0,
    "random_state": 42,
}

# These are the hyper tuned parameters for XGBoost.
xgboost_params = {
    "n_estimators": 2000,
    "learning_rate": 0.04837313058498289,
    "max_depth": 9,
    "min_child_weight": 4,
    "subsample": 0.9180377209506578,
    "colsample_bytree": 0.7030304308442571,
    "gamma": 0.11226802436236458,
    "reg_alpha": 0.0002552479075743287,
    "reg_lambda": 3.099565353271576,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

'''
# This prints the CatBoost parameters.
print("\nCatBoost")
print("Hyperparameters:")
for k, v in catboost_params.items():
    print(f"  {k}: {v}")

# This prints the XGBoost parameters.
print("\nXGBoost")
print("Hyperparameters:")
for k, v in xgboost_params.items():
    print(f"  {k}: {v}")

    '''
    
# This first trains the two base models on the base-train split.
cat_model = CatBoostRegressor(**catboost_params)
cat_model.fit(X_base_train_cat, y_base_train, verbose=False)
xgb_model = XGBRegressor(**xgboost_params)
xgb_model.fit(X_base_train_xgb, y_base_train)

# This gets predictions needed for the second layer.
meta_valid_cat_pred = cat_model.predict(X_meta_valid_cat)
meta_valid_xgb_pred = xgb_model.predict(X_meta_valid_xgb)
meta_X_valid = pd.DataFrame({
    "cat_pred": meta_valid_cat_pred,
    "xgb_pred": meta_valid_xgb_pred, })

# This trains the  second layer (linear stacking model)
meta_model = LinearRegression()
meta_model.fit(meta_X_valid, y_meta_valid)

print("\nLinear layer")
print("Weights learned:")
print(f"  CatBoost weight: {meta_model.coef_[0]:.6f}")
print(f"  XGBoost weight : {meta_model.coef_[1]:.6f}")
print(f"  Intercept      : {meta_model.intercept_:.6f}")

# This retrains the base models on the full training set.
cat_model_full = CatBoostRegressor(**catboost_params)
cat_model_full.fit(X_train_model, y_train, verbose=False)

xgb_model_full = XGBRegressor(**xgboost_params)
xgb_model_full.fit(X_train_safe, y_train)

# This makes predictions on the test set.
y_pred_cat = cat_model_full.predict(X_test_model)
y_pred_xgb = xgb_model_full.predict(X_test_safe)
y_pred_avg = (y_pred_cat + y_pred_xgb) / 2.0

meta_X_test = pd.DataFrame({
    "cat_pred": y_pred_cat,
    "xgb_pred": y_pred_xgb, })
y_pred_stack = meta_model.predict(meta_X_test)

# This evaluates each prediction method.
cat_rmse, cat_mae, cat_r2 = evaluate_regression(y_test, y_pred_cat)
xgb_rmse, xgb_mae, xgb_r2 = evaluate_regression(y_test, y_pred_xgb)
avg_rmse, avg_mae, avg_r2 = evaluate_regression(y_test, y_pred_avg)
stack_rmse, stack_mae, stack_r2 = evaluate_regression(y_test, y_pred_stack)

# This prints the CatBoost test results.
print("\nCatBoost only")
print(f"  RMSE: {cat_rmse:.6f}")
print(f"  MAE : {cat_mae:.6f}")
print(f"  R2  : {cat_r2:.6f}")

# This prints the XGBoost test results.
print("\nXGBoost only")
print(f"  RMSE: {xgb_rmse:.6f}")
print(f"  MAE : {xgb_mae:.6f}")
print(f"  R2  : {xgb_r2:.6f}")

# This prints the simple average blend results.
print("\nAverage blend")
print(f"  RMSE: {avg_rmse:.6f}")
print(f"  MAE : {avg_mae:.6f}")
print(f"  R2  : {avg_r2:.6f}")

# This prints the stacked linear layer results.
print("\nStacked linear layer")
print(f"  RMSE: {stack_rmse:.6f}")
print(f"  MAE : {stack_mae:.6f}")
print(f"  R2  : {stack_r2:.6f}")

# This builds a final comparison table.
results_df = pd.DataFrame(
    [
        {"Model": "CatBoost only", "RMSE": cat_rmse, "MAE": cat_mae, "R2": cat_r2},
        {"Model": "XGBoost only", "RMSE": xgb_rmse, "MAE": xgb_mae, "R2": xgb_r2},
        {"Model": "Average blend", "RMSE": avg_rmse, "MAE": avg_mae, "R2": avg_r2},
        {"Model": "Stacked linear layer", "RMSE": stack_rmse, "MAE": stack_mae, "R2": stack_r2},
    ]
).sort_values("R2", ascending=False).reset_index(drop=True)

# This prints the ranked model table.
print("\nModel comparison:")
print(results_df.to_string(index=False))