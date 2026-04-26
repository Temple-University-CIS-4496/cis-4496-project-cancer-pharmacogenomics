
import matplotlib

from Main_Preprocessing_2 import df_original_clean

print(df_original_clean.head())

import matplotlib.pyplot as plt

print("EDA script is running...")

cancer_types = df_original_clean["TCGA_DESC"].unique()

for cancer in cancer_types:
    subset = df_original_clean[df_original_clean["TCGA_DESC"] == cancer]
    drug_col = "DRUG_NAME"
    ic50_col = "LN_IC50"
    top_drugs = subset[drug_col].value_counts().nlargest(10).index
    subset = subset[subset[drug_col].isin(top_drugs)]

    avg_ic50 = subset.groupby(drug_col)[ic50_col].mean().sort_values()

    plt.figure()
    plt.bar(avg_ic50.index, avg_ic50.values)

    plt.xticks(rotation=45)
    plt.title(f"{cancer} - Drug Sensitivity")
    plt.tight_layout()

    #plt.savefig(f"{cancer}_IC50_plot.png")  # saves instead of showing
    plt.close()
    matplotlib.pyplot.close()

df_plot = df_original_clean.copy()

# Optional: limit cancers so you don’t get 50 plots

# limit cancers
top_cancers = df_plot["TCGA_DESC"].value_counts().nlargest(5).index
df_small = df_plot[df_plot["TCGA_DESC"].isin(top_cancers)]

# limit drugs
top_drugs = df_small["DRUG_NAME"].value_counts().nlargest(10).index
df_small = df_small[df_small["DRUG_NAME"].isin(top_drugs)]

pivot = df_small.pivot_table(
    values="LN_IC50",
    index="TCGA_DESC",
    columns="DRUG_NAME",
    aggfunc="mean"
)

plt.figure()
plt.imshow(pivot, aspect='auto')
plt.colorbar(label="Mean IC50")

plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
plt.yticks(range(len(pivot.index)), pivot.index)

plt.title("IC50 Heatmap (Top Cancers & Drugs)")
plt.tight_layout()



drug_variance = df_original_clean.groupby("DRUG_NAME")["LN_IC50"].var()
top_variable_drugs = drug_variance.nlargest(15).index

df_var = df_original_clean[df_original_clean["DRUG_NAME"].isin(top_variable_drugs)]

pivot_var = df_var.pivot_table(
    values="LN_IC50",
    index="TCGA_DESC",
    columns="DRUG_NAME",
    aggfunc="mean"
)

plt.figure(figsize=(12, 8))
plt.imshow(pivot_var, aspect='auto', cmap='plasma')
plt.colorbar(label="Mean IC50")

plt.xticks(range(len(pivot_var.columns)), pivot_var.columns, rotation=45, ha='right')
plt.yticks(range(len(pivot_var.index)), pivot_var.index)

plt.title("IC50 Heatmap (All Cancers, Most Informative Drugs)")
plt.tight_layout()




import seaborn as sns
import matplotlib.pyplot as plt

df_filtered = df_small[~df_small["TCGA_DESC"].str.lower().str.contains("unclassified", na=False)]
top_drugs = df_filtered["DRUG_NAME"].value_counts().nlargest(5).index
df_plot2 = df_filtered[df_filtered["DRUG_NAME"].isin(top_drugs)]
plt.figure(figsize=(12, 6))
sns.boxplot(
    data= df_plot2,
    x="TCGA_DESC", 
    y="LN_IC50",
    hue="DRUG_NAME"
)

plt.title("IC50 Distribution by Cancer Type")
plt.xticks(rotation=45)

plt.tight_layout()

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.5)

# perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel("Actual LN_IC50")
plt.ylabel("Predicted LN_IC50")
plt.title("Predicted vs Actual (Model Performance)")

plt.grid(True)
plt.tight_layout()
plt.show()