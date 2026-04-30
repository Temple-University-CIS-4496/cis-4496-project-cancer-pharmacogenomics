import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Cancer Pharmacogenomics Dashboard",
    page_icon="🧬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PREPROCESS_FILE = BASE_DIR / "Main_Preprocessing_2.py"
MODELING_FILE = BASE_DIR / "Main_modeling_3.py"
GUIDE_FILE = BASE_DIR / "GUIDE_INFO.md"


@st.cache_resource(show_spinner=True)
def load_pipeline_context():
    shared = {
        "__name__": "__main__",
        "__file__": str(PREPROCESS_FILE),
    }

    original_cwd = os.getcwd()
    try:
        os.chdir(BASE_DIR)

        with open(PREPROCESS_FILE, "r", encoding="utf-8") as f:
            preprocess_code = f.read()
        exec(preprocess_code, shared)

        shared["__file__"] = str(MODELING_FILE)
        with open(MODELING_FILE, "r", encoding="utf-8") as f:
            modeling_code = f.read()
        exec(modeling_code, shared)

    finally:
        os.chdir(original_cwd)

    return shared


@st.cache_data(show_spinner=False)
def load_guide_text():
    if GUIDE_FILE.exists():
        return GUIDE_FILE.read_text(encoding="utf-8")
    return "Guide file not found. Add GUIDE_INFO.md to the same folder as this dashboard."


def sidebar_select_from_column(label, df_source, column_name, current_value):
    options = sorted(df_source[column_name].dropna().astype(str).unique().tolist())
    if current_value not in options:
        options = [current_value] + options
    return st.sidebar.selectbox(label, options, index=options.index(current_value))


def sidebar_slider_from_column(df_source, raw_profile, column_name, step=0.05):
    col_series = pd.to_numeric(df_source[column_name], errors="coerce").dropna()
    current_val = float(pd.to_numeric(raw_profile[column_name].iloc[0], errors="coerce"))
    min_val = float(col_series.min())
    max_val = float(col_series.max())

    return st.sidebar.slider(
        column_name,
        min_value=min_val,
        max_value=max_val,
        value=float(np.clip(current_val, min_val, max_val)),
        step=step,
    )


def compare_profiles(original_profile, edited_profile, cols_to_check):
    changed_fields = []
    for col in cols_to_check:
        if col in original_profile.columns and col in edited_profile.columns:
            old_val = str(original_profile[col].iloc[0])
            new_val = str(edited_profile[col].iloc[0])
            if old_val != new_val:
                changed_fields.append(
                    {
                        "Field": col,
                        "Original": old_val,
                        "Edited": new_val,
                    }
                )
    return pd.DataFrame(changed_fields)


def show_table(df):
    st.dataframe(df, width="stretch")


def build_single_row_for_models(raw_row_df, ctx):
    row = raw_row_df.copy()

    candidate_feature_cols = ctx["candidate_feature_cols"]
    row = row[candidate_feature_cols].copy()

    if "DRUG_NAME" in row.columns:
        row["DRUG_NAME"] = row["DRUG_NAME"].astype("string").fillna("MISSING")

        if "drug_te_map" in ctx:
            drug_map = ctx["drug_te_map"]
        elif "drug_target_map" in ctx:
            drug_map = ctx["drug_target_map"]
        elif "drug_mean_map" in ctx:
            drug_map = ctx["drug_mean_map"]
        else:
            raise KeyError(
                "No DRUG_NAME encoding map found. Expected one of: "
                "'drug_te_map', 'drug_target_map', or 'drug_mean_map'."
            )

        if "global_train_mean" in ctx:
            global_train_mean = float(ctx["global_train_mean"])
        elif "y_train" in ctx:
            global_train_mean = float(ctx["y_train"].mean())
        else:
            raise KeyError("Could not find global_train_mean or y_train in context.")

        row["DRUG_NAME_target_enc"] = row["DRUG_NAME"].map(drug_map).fillna(global_train_mean)
        row = row.drop(columns=["DRUG_NAME"])

    ohe_cols = ctx["ohe_cols"]
    numeric_cols = ctx["numeric_cols"]
    train_medians = ctx["train_medians"]

    for col in ohe_cols:
        if col in row.columns:
            row[col] = row[col].astype("string").fillna("MISSING")

    for col in numeric_cols:
        if col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce")

    row[numeric_cols] = row[numeric_cols].fillna(train_medians)

    row_cat = pd.get_dummies(row[ohe_cols], columns=ohe_cols, dummy_na=False)

    training_encoded_cols = ctx["X_train_encoded"].columns.tolist()
    row_encoded = pd.concat([row_cat, row[numeric_cols]], axis=1)
    row_encoded = row_encoded.reindex(columns=training_encoded_cols, fill_value=0)

    numeric_cols_to_scale = ctx["numeric_cols_to_scale"]

    if "quantile_scaler" in ctx:
        transformer = ctx["quantile_scaler"]
    elif "scaler" in ctx:
        transformer = ctx["scaler"]
    else:
        raise KeyError("Could not find 'quantile_scaler' or 'scaler' in context.")

    row_scaled = row_encoded.copy()
    row_scaled[numeric_cols_to_scale] = transformer.transform(
        row_encoded[numeric_cols_to_scale]
    )

    feature_map = ctx["feature_map"]
    row_xgb = row_scaled.copy()
    row_xgb.columns = feature_map["safe_name"].tolist()

    row_catboost = row_scaled.copy()

    return row_catboost, row_xgb


def predict_profile(raw_row_df, ctx):
    row_catboost, row_xgb = build_single_row_for_models(raw_row_df, ctx)

    cat_pred = float(ctx["cat_model_full"].predict(row_catboost)[0])
    xgb_pred = float(ctx["xgb_model_full"].predict(row_xgb)[0])
    avg_pred = float((cat_pred + xgb_pred) / 2.0)

    meta_X = pd.DataFrame(
        {
            "cat_pred": [cat_pred],
            "xgb_pred": [xgb_pred],
        }
    )
    stack_pred = float(ctx["meta_model"].predict(meta_X)[0])

    preds = {
        "CatBoost only": cat_pred,
        "XGBoost only": xgb_pred,
        "Average blend": avg_pred,
        "Stacked linear layer": stack_pred,
    }

    pred_mean = float(np.mean(list(preds.values())))
    pred_std = float(np.std(list(preds.values())))

    return preds, pred_mean, pred_std


def get_prediction_strength(raw_row_df, ctx, pred_std):
    df6 = ctx["df6"]

    tcga_val = raw_row_df["TCGA_DESC"].iloc[0] if "TCGA_DESC" in raw_row_df.columns else None
    drug_val = raw_row_df["DRUG_NAME"].iloc[0] if "DRUG_NAME" in raw_row_df.columns else None

    tcga_count = int((df6["TCGA_DESC"] == tcga_val).sum()) if tcga_val is not None else 0
    drug_count = int((df6["DRUG_NAME"] == drug_val).sum()) if drug_val is not None else 0

    if pred_std < 0.08 and tcga_count >= 100 and drug_count >= 100:
        return "High", tcga_count, drug_count
    elif pred_std < 0.20 and tcga_count >= 25 and drug_count >= 25:
        return "Moderate", tcga_count, drug_count
    else:
        return "Low", tcga_count, drug_count


def rank_drugs_for_profile(raw_row_df, ctx, max_drugs=15):
    df6 = ctx["df6"]

    common_drugs = (
        df6["DRUG_NAME"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(max_drugs)
        .index
        .tolist()
    )

    ranking_rows = []

    for drug_name in common_drugs:
        temp_row = raw_row_df.copy()
        temp_row["DRUG_NAME"] = drug_name

        _, pred_mean, pred_std = predict_profile(temp_row, ctx)
        strength, _, _ = get_prediction_strength(temp_row, ctx, pred_std)

        ranking_rows.append(
            {
                "Drug": drug_name,
                "Predicted_ln_IC50": pred_mean,
                "Prediction_Strength": strength,
                "Method_Spread_STD": pred_std,
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        "Predicted_ln_IC50",
        ascending=True,
    ).reset_index(drop=True)

    ranking_df.insert(0, "Rank", np.arange(1, len(ranking_df) + 1))
    return ranking_df


@st.cache_data(show_spinner=False)
def build_feature_importance_df(_ctx):
    feature_map = _ctx["feature_map"].copy()

    cat_importance = pd.DataFrame(
        {
            "original_name": _ctx["X_train_model"].columns,
            "catboost_importance": _ctx["cat_model_full"].feature_importances_,
        }
    )

    xgb_importance = pd.DataFrame(
        {
            "safe_name": feature_map["safe_name"],
            "xgboost_importance": _ctx["xgb_model_full"].feature_importances_,
        }
    ).merge(feature_map, on="safe_name", how="left")

    merged = cat_importance.merge(
        xgb_importance[["original_name", "xgboost_importance"]],
        on="original_name",
        how="outer",
    ).fillna(0)

    cat_max = merged["catboost_importance"].max()
    xgb_max = merged["xgboost_importance"].max()

    merged["catboost_norm"] = (
        merged["catboost_importance"] / cat_max if cat_max > 0 else 0
    )
    merged["xgboost_norm"] = (
        merged["xgboost_importance"] / xgb_max if xgb_max > 0 else 0
    )
    merged["combined_importance"] = (
        merged["catboost_norm"] + merged["xgboost_norm"]
    ) / 2.0

    merged = merged.sort_values("combined_importance", ascending=False).reset_index(drop=True)
    return merged


def get_matched_subset(df, selected_drug, selected_tcga):
    subset = df.copy()
    source_label = "Selected drug"

    if selected_drug is not None and "DRUG_NAME" in subset.columns:
        subset = subset[subset["DRUG_NAME"].astype(str) == str(selected_drug)].copy()

    if len(subset) >= 20:
        return subset, source_label

    if selected_tcga is not None and "TCGA_DESC" in df.columns:
        subset = df[df["TCGA_DESC"].astype(str) == str(selected_tcga)].copy()
        source_label = "Selected TCGA_DESC"
        return subset, source_label

    return df.copy(), "Full dataset fallback"


def calculate_subset_stats(subset_df, value_col, current_prediction, original_actual):
    values = pd.to_numeric(subset_df[value_col], errors="coerce").dropna()

    if len(values) == 0:
        return {}, pd.DataFrame()

    q1 = float(values.quantile(0.25))
    median = float(values.median())
    q3 = float(values.quantile(0.75))
    iqr = float(q3 - q1)
    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)
    outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())

    prediction_percentile = float((values <= current_prediction).mean() * 100)
    actual_percentile = float((values <= original_actual).mean() * 100)

    stats_dict = {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "median": median,
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "prediction_percentile": prediction_percentile,
        "actual_percentile": actual_percentile,
        "prediction_vs_mean": float(current_prediction - values.mean()),
        "prediction_vs_median": float(current_prediction - median),
        "prediction_inside_iqr_range": bool(lower_bound <= current_prediction <= upper_bound),
    }

    stats_table = pd.DataFrame(
        [
            {"Statistic": "Matched subset size", "Value": stats_dict["count"]},
            {"Statistic": "Mean observed LN_IC50", "Value": round(stats_dict["mean"], 4)},
            {"Statistic": "Median observed LN_IC50", "Value": round(stats_dict["median"], 4)},
            {"Statistic": "Standard deviation", "Value": round(stats_dict["std"], 4)},
            {"Statistic": "Minimum", "Value": round(stats_dict["min"], 4)},
            {"Statistic": "Maximum", "Value": round(stats_dict["max"], 4)},
            {"Statistic": "Q1", "Value": round(stats_dict["q1"], 4)},
            {"Statistic": "Q3", "Value": round(stats_dict["q3"], 4)},
            {"Statistic": "IQR", "Value": round(stats_dict["iqr"], 4)},
            {"Statistic": "Lower outlier bound", "Value": round(stats_dict["lower_bound"], 4)},
            {"Statistic": "Upper outlier bound", "Value": round(stats_dict["upper_bound"], 4)},
            {"Statistic": "Outlier count", "Value": stats_dict["outlier_count"]},
            {"Statistic": "Current prediction percentile", "Value": round(stats_dict["prediction_percentile"], 2)},
            {"Statistic": "Original actual percentile", "Value": round(stats_dict["actual_percentile"], 2)},
            {"Statistic": "Prediction minus subset mean", "Value": round(stats_dict["prediction_vs_mean"], 4)},
            {"Statistic": "Prediction minus subset median", "Value": round(stats_dict["prediction_vs_median"], 4)},
        ]
    )

    return stats_dict, stats_table


def build_distribution_df(df, value_col, bins=20):
    values = pd.to_numeric(df[value_col], errors="coerce").dropna().values
    counts, edges = np.histogram(values, bins=bins)

    return pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "count": counts,
        }
    )


def make_bar_chart(df, x_col, y_col, color_col=None, title=""):
    if color_col:
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(x_col, sort=None),
                y=alt.Y(y_col),
                color=alt.Color(color_col, legend=None),
                tooltip=list(df.columns),
            )
            .properties(height=360, title=title)
        )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x_col, sort=None),
            y=alt.Y(y_col),
            tooltip=list(df.columns),
        )
        .properties(height=360, title=title)
    )


with st.spinner("Running preprocessing and modeling files..."):
    ctx = load_pipeline_context()

df6 = ctx["df6"].copy()
results_df = ctx["results_df"].copy()
y_test = ctx["y_test"].copy()
guide_text = load_guide_text()
importance_df = build_feature_importance_df(ctx)

test_indices = y_test.index.tolist()

display_cols = [
    "DRUG_NAME",
    "TCGA_DESC",
    "Microsatellite instability Status (MSI)",
    "Growth Properties",
    "mutational_burden",
    "ploidy_snp6",
    "ploidy_wes",
    "LN_IC50",
]
display_cols = [c for c in display_cols if c in df6.columns]

st.title("Drug Response Dashboard")
st.markdown(
    """
This dashboard connects the updated preprocessing and hybrid modeling pipeline to an interactive view of predictions, rankings, comparisons, statistics, and charts.
"""
)
st.write("This dashboard lets you explore how the trained models estimate drug sensitivity for a selected cancer-related profile.")

if "show_guide" not in st.session_state:
    st.session_state.show_guide = False

top_left, top_right = st.columns([20, 1])

with top_right:
    if st.button("G", help="Open the dashboard guide"):
        st.session_state.show_guide = not st.session_state.show_guide

if st.session_state.show_guide:
    st.subheader("Dashboard Guide")
    st.write("This guide explains what the inputs mean and how each dashboard section should be interpreted.")
    st.markdown(guide_text)

st.sidebar.header("Select a Test Profile")
st.sidebar.caption("Choose a held-out sample, then adjust a small set of profile features to test how predictions change.")
selected_index = st.sidebar.selectbox(
    "Choose held-out test sample",
    options=test_indices,
    index=0,
)

raw_profile = df6.loc[[selected_index]].copy()
original_profile = df6.loc[[selected_index]].copy()

st.sidebar.header("What-If Inputs")
st.sidebar.caption("These controls let you modify the selected profile so the dashboard can compare the new prediction against the original one.")

drug_options = (
    df6["DRUG_NAME"].dropna().astype(str).value_counts().head(30).index.tolist()
)
default_drug = str(raw_profile["DRUG_NAME"].iloc[0]) if "DRUG_NAME" in raw_profile.columns else drug_options[0]
if default_drug not in drug_options:
    drug_options = [default_drug] + drug_options
selected_drug = st.sidebar.selectbox("Drug", drug_options, index=drug_options.index(default_drug))
raw_profile["DRUG_NAME"] = selected_drug

if "TCGA_DESC" in raw_profile.columns:
    current_tcga = str(raw_profile["TCGA_DESC"].iloc[0])
    raw_profile["TCGA_DESC"] = sidebar_select_from_column("TCGA_DESC", df6, "TCGA_DESC", current_tcga)

msi_col = "Microsatellite instability Status (MSI)"
if msi_col in raw_profile.columns:
    current_msi = str(raw_profile[msi_col].iloc[0])
    raw_profile[msi_col] = sidebar_select_from_column("MSI Status", df6, msi_col, current_msi)

growth_col = "Growth Properties"
if growth_col in raw_profile.columns:
    current_growth = str(raw_profile[growth_col].iloc[0])
    raw_profile[growth_col] = sidebar_select_from_column("Growth Properties", df6, growth_col, current_growth)

for num_col in ["mutational_burden", "ploidy_snp6", "ploidy_wes"]:
    if num_col in raw_profile.columns:
        raw_profile[num_col] = sidebar_slider_from_column(df6, raw_profile, num_col)

run_button = st.sidebar.button("Run Prediction")

preds, pred_mean, pred_std = predict_profile(raw_profile, ctx)
strength_label, tcga_count, drug_count = get_prediction_strength(raw_profile, ctx, pred_std)
ranking_df = rank_drugs_for_profile(raw_profile, ctx, max_drugs=15)
original_preds, original_pred_mean, _ = predict_profile(original_profile, ctx)

pred_df = pd.DataFrame(
    {
        "Model_Method": list(preds.keys()),
        "Predicted_ln_IC50": list(preds.values()),
    }
).sort_values("Predicted_ln_IC50", ascending=True).reset_index(drop=True)

compare_df = pd.DataFrame(
    [
        {
            "Scenario": "Original held-out row",
            "Average_Predicted_ln_IC50": original_pred_mean,
        },
        {
            "Scenario": "Edited dashboard profile",
            "Average_Predicted_ln_IC50": pred_mean,
        },
    ]
)

changed_fields_df = compare_profiles(
    original_profile,
    raw_profile,
    ["DRUG_NAME", "TCGA_DESC", msi_col, growth_col, "mutational_burden", "ploidy_snp6", "ploidy_wes"],
)

strength_details = pd.DataFrame(
    [
        {"Signal": "Method agreement (STD across prediction methods)", "Value": round(pred_std, 4)},
        {"Signal": "Rows with selected TCGA_DESC", "Value": tcga_count},
        {"Signal": "Rows with selected DRUG_NAME", "Value": drug_count},
    ]
)

top_importance_df = importance_df.head(15).copy()

selected_drug_value = str(raw_profile["DRUG_NAME"].iloc[0]) if "DRUG_NAME" in raw_profile.columns else None
selected_tcga_value = str(raw_profile["TCGA_DESC"].iloc[0]) if "TCGA_DESC" in raw_profile.columns else None

matched_subset, subset_source_label = get_matched_subset(df6, selected_drug_value, selected_tcga_value)
matched_stats, stats_table = calculate_subset_stats(
    matched_subset,
    "LN_IC50",
    pred_mean,
    float(df6.loc[selected_index, "LN_IC50"]),
)

st.write("These summary metrics show the current average prediction, how much it changed, how reliable it looks, and how closely the prediction methods agree.")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Average Predicted ln(IC50)", f"{pred_mean:.4f}")

with metric_col2:
    delta_val = pred_mean - original_pred_mean
    st.metric("What-If Change vs Original", f"{pred_mean:.4f}", delta=f"{delta_val:.4f}")

with metric_col3:
    st.metric("Prediction Strength", strength_label)

with metric_col4:
    st.metric("Method Agreement STD", f"{pred_std:.4f}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Overview",
        "Model Comparison",
        "Drug Ranking",
        "What-If Analysis",
        "Prediction Strength",
        "Feature Influence",
        "Stats",
        "Charts",
    ]
)

with tab1:
    st.subheader("Overview")
    st.write("This section gives a quick summary of the model results and the profile currently being tested.")
    st.caption("The table below shows how the current modeling methods performed overall.")
    show_table(results_df)

    st.caption("The table below shows the raw held-out profile that is feeding the current dashboard prediction.")
    show_table(raw_profile[display_cols])

    st.info(
        "This dashboard reruns the updated preprocessing and hybrid modeling scripts first, "
        "then uses the trained models and intermediate objects inside the interface."
    )

with tab2:
    st.subheader("Model Comparison")
    st.write("This section compares how each prediction method scores the same selected profile so you can see agreement and differences across models.")
    st.caption("The table below lists each method's predicted LN_IC50 for the current profile.")
    show_table(pred_df)

    st.write("Average predicted ln(IC50):", round(pred_mean, 4))
    st.write("Actual LN_IC50 for original held-out row:", round(float(df6.loc[selected_index, "LN_IC50"]), 4))
    st.info("See the Charts tab for the visual comparison of these prediction methods.")

with tab3:
    st.subheader("Drug Ranking")
    st.write("This section ranks common drugs for the selected profile so you can compare which options appear more or less favorable.")
    st.caption("The table below shows the ranked drugs and their predicted LN_IC50 values for the selected profile.")
    show_table(ranking_df)
    st.info("See the Charts tab for the visual ranking of the top drugs and the relative change from the selected drug.")

with tab4:
    st.subheader("What-If Analysis")
    st.write("This section compares the original held-out profile with your edited version to show how the prediction changes after modifying the inputs.")
    show_table(compare_df)

    if not changed_fields_df.empty:
        st.caption("The table below lists exactly which profile fields were changed in the sidebar.")
        show_table(changed_fields_df)
    else:
        st.write("No fields changed yet. You are still viewing the original held-out profile.")

    st.info("See the Charts tab for the visual original-versus-edited comparison.")

with tab5:
    st.subheader("Prediction Strength")
    st.write("This section shows how well-supported the current prediction is based on method agreement and how much related data exists in the dataset.")
    st.write("Current strength label:", strength_label)
    st.caption("The values below explain why the current prediction strength label was assigned.")
    show_table(strength_details)

    st.warning(
        "This is a draft reliability indicator for the capstone dashboard. "
        "It is not a formal uncertainty interval or clinical confidence score."
    )
    st.info("See the Charts tab for the visual support chart.")

with tab6:
    st.subheader("Feature Influence")
    st.write("This section highlights which encoded features appear most influential across the main base models used in the dashboard.")
    st.caption("The table below shows the strongest features based on normalized importance across CatBoost and XGBoost.")
    show_table(top_importance_df[["original_name", "catboost_importance", "xgboost_importance", "combined_importance"]])
    st.info("See the Charts tab for the visual feature-importance view.")

with tab7:
    st.subheader("Stats")
    st.write("This section summarizes how the current prediction compares to a related observed subset from the dataset.")

    current_summary_df = pd.DataFrame(
        [
            {"Statistic": "Subset source", "Value": subset_source_label},
            {"Statistic": "Current average prediction", "Value": round(pred_mean, 4)},
            {"Statistic": "Original held-out actual LN_IC50", "Value": round(float(df6.loc[selected_index, "LN_IC50"]), 4)},
            {"Statistic": "Prediction strength", "Value": strength_label},
            {"Statistic": "Method agreement STD", "Value": round(pred_std, 4)},
        ]
    )
    st.caption("The table below summarizes the current selected case and its prediction outputs.")
    show_table(current_summary_df)

    if matched_stats:
        st.caption("The table below summarizes the matched observed subset used to provide context for the current prediction.")
        show_table(stats_table)

        interpretation_df = pd.DataFrame(
            [
                {
                    "Interpretation": "Current prediction inside IQR range",
                    "Value": "Yes" if matched_stats["prediction_inside_iqr_range"] else "No",
                },
                {
                    "Interpretation": "Current prediction percentile",
                    "Value": f"{matched_stats['prediction_percentile']:.2f}",
                },
                {
                    "Interpretation": "Original actual percentile",
                    "Value": f"{matched_stats['actual_percentile']:.2f}",
                },
                {
                    "Interpretation": "Prediction minus subset mean",
                    "Value": round(matched_stats["prediction_vs_mean"], 4),
                },
                {
                    "Interpretation": "Prediction minus subset median",
                    "Value": round(matched_stats["prediction_vs_median"], 4),
                },
            ]
        )
        st.caption("The table below explains where the current prediction sits relative to similar observed values.")
        show_table(interpretation_df)

    else:
        st.warning("No valid LN_IC50 values were found for the matched subset statistics.")

with tab8:
    st.subheader("Charts")
    st.write("This section gathers the visual versions of the main analyses so the dashboard stays cleaner in the other tabs.")

    st.markdown("**Model Comparison Chart**")
    method_chart = make_bar_chart(
        pred_df,
        x_col="Model_Method:N",
        y_col="Predicted_ln_IC50:Q",
        title="Prediction by model method",
    )
    st.altair_chart(method_chart, use_container_width=True)

    st.markdown("**Drug Ranking Chart**")
    ranking_top_df = ranking_df.head(10).copy()
    ranking_chart = make_bar_chart(
        ranking_top_df,
        x_col="Drug:N",
        y_col="Predicted_ln_IC50:Q",
        title="Top ranked drugs for this selected profile",
    )
    st.altair_chart(ranking_chart, use_container_width=True)

    st.markdown("**What-If Comparison Chart**")
    what_if_chart = make_bar_chart(
        compare_df,
        x_col="Scenario:N",
        y_col="Average_Predicted_ln_IC50:Q",
        title="Original versus edited prediction",
    )
    st.altair_chart(what_if_chart, use_container_width=True)

    st.markdown("**Prediction Strength Support Chart**")
    strength_support_df = pd.DataFrame(
        [
            {"Signal": "Rows with selected TCGA_DESC", "Value": tcga_count},
            {"Signal": "Rows with selected DRUG_NAME", "Value": drug_count},
        ]
    )
    strength_chart = make_bar_chart(
        strength_support_df,
        x_col="Signal:N",
        y_col="Value:Q",
        title="How much data supports this profile",
    )
    st.altair_chart(strength_chart, use_container_width=True)

    st.markdown("**Feature Influence Chart**")
    feature_chart = make_bar_chart(
        top_importance_df,
        x_col="original_name:N",
        y_col="combined_importance:Q",
        title="Most influential features across the base models",
    )
    st.altair_chart(feature_chart, use_container_width=True)

    st.markdown("**Matched Subset Distribution Histogram**")
    hist_df = build_distribution_df(matched_subset, "LN_IC50", bins=20)
    hist_chart = (
        alt.Chart(hist_df)
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title="LN_IC50"),
            x2="bin_end:Q",
            y=alt.Y("count:Q", title="Count"),
            tooltip=["bin_start", "bin_end", "count"],
        )
        .properties(height=360, title=f"Observed LN_IC50 distribution from: {subset_source_label}")
    )

    pred_rule_df = pd.DataFrame(
        {
            "label": ["Current average prediction", "Original actual LN_IC50"],
            "value": [pred_mean, float(df6.loc[selected_index, "LN_IC50"])],
        }
    )

    rule_chart = (
        alt.Chart(pred_rule_df)
        .mark_rule(size=3)
        .encode(
            x="value:Q",
            color="label:N",
            tooltip=["label", "value"],
        )
    )

    distribution_chart = hist_chart + rule_chart
    st.altair_chart(distribution_chart, use_container_width=True)

    st.markdown("**Matched Subset Box Plot**")
    box_input_df = matched_subset[["LN_IC50"]].copy()
    box_input_df["Group"] = subset_source_label
    box_chart = (
        alt.Chart(box_input_df)
        .mark_boxplot()
        .encode(
            x=alt.X("Group:N", title="Matched subset"),
            y=alt.Y("LN_IC50:Q", title="LN_IC50"),
            tooltip=["Group", "LN_IC50"],
        )
        .properties(height=360, title="Matched subset LN_IC50 box plot")
    )
    st.altair_chart(box_chart, use_container_width=True)

    st.markdown("**Prediction Position Chart**")
    if matched_stats:
        position_df = pd.DataFrame(
            [
                {"Metric": "Min", "Value": matched_stats["min"]},
                {"Metric": "Q1", "Value": matched_stats["q1"]},
                {"Metric": "Median", "Value": matched_stats["median"]},
                {"Metric": "Q3", "Value": matched_stats["q3"]},
                {"Metric": "Max", "Value": matched_stats["max"]},
                {"Metric": "Current prediction", "Value": pred_mean},
                {"Metric": "Original actual", "Value": float(df6.loc[selected_index, "LN_IC50"])},
            ]
        )

        position_chart = (
            alt.Chart(position_df)
            .mark_point(size=120)
            .encode(
                x=alt.X("Value:Q", title="LN_IC50"),
                y=alt.Y("Metric:N", sort=["Min", "Q1", "Median", "Q3", "Max", "Current prediction", "Original actual"]),
                tooltip=["Metric", "Value"],
            )
            .properties(height=320, title="Where the current prediction sits within the matched subset")
        )
        st.altair_chart(position_chart, use_container_width=True)

    st.markdown("**Ranked Drug Delta Chart**")
    delta_reference = pred_mean
    ranking_delta_df = ranking_df.head(10).copy()
    ranking_delta_df["Delta_vs_Selected"] = ranking_delta_df["Predicted_ln_IC50"] - delta_reference
    delta_chart = make_bar_chart(
        ranking_delta_df,
        x_col="Drug:N",
        y_col="Delta_vs_Selected:Q",
        title="Difference from the currently selected drug prediction",
    )
    st.altair_chart(delta_chart, use_container_width=True)

if run_button:
    st.success("Prediction updated using the selected dashboard profile.")