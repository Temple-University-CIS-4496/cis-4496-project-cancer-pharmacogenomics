# Final Model Report
_Report describing the final model to be delivered - typically comprised of one or more of the models built during the life of the project_

## Analytic Approach
* What is target definition
* What are inputs (description)
* What kind of model was built?

The primary target variable for this project is LN_IC50, defined as the natural logarithm of the half-maximal inhibitory concentration. LN_IC50 was selected because it provides the best balance of statistical dispersion, biological interpretability, and alignment with established GDSC drug-response endpoints, compared to alternatives such as AUC and Z-scores. The modeling task is framed as a supervised regression problem, with LN_IC50 treated as a continuous outcome, enabling precise prediction of drug sensitivity across diverse cancer cell lines.

The model inputs were constructed from a carefully engineered set of biological, genomic, tissue, and drug-related features. The final modeling dataset includes drug identity (DRUG_NAME), mechanism-level information describing drug targets (TARGET_PATHWAY), standardized cancer classifications (TCGA_DESC), and tissue-level descriptors (GDSC Tissue descriptor 1 and GDSC Tissue descriptor 2). These are complemented by phenotypic indicators such as microsatellite instability (MSI) and growth properties, as well as genomic predictors including mutational_burden and ploidy estimates derived from SNP array and whole-exome sequencing (ploidy_snp6 and ploidy_wes). The feature engineering pipeline included imputing missing values, one-hot encoding low-cardinality categorical variables, target encoding high-cardinality features such as DRUG_NAME, and standardization of continuous predictors. In addition, feature selection was supported using mutual information analysis and repeated subsampling to retain the most informative predictors.

The analytic strategy initially evaluated a broad set of models, including linear, tree-based, and neural network approaches. Then, it focused on gradient boosting methods, such as CatBoost, XGBoost, and LightGBM, which were the most effective at capturing the nonlinear relationships and feature interactions inherent in drug response data. To improve methodological rigor, the pipeline incorporated cross-validation target encoding. This ensured that encoded features were generated out-of-fold, reducing data leakage and producing more reliable performance estimates. Although this stricter encoding initially resulted in slightly lower performance, subsequent hyperparameter tuning restored and improved model effectiveness.

The final modeling approach used a stacked hybrid architecture that combined a CatBoost and an XGBoost as base models with a second-layer linear regression model. This design leverages the complementary strengths of the models and improves predictive performance beyond that of the individual models. Model performance was evaluated using R^2, RMSE, and MAE, which provide a balanced assessment of explained variance, sensitivity to large errors, and typical prediction accuracy.

## Solution Description
* Simple solution architecture (Data sources, solution components, data flow)
* What is output?

## Data
* Source
* Data Schema
* Sampling
* Selection (dates, segments)
* Stats (counts)

## Features
* List of raw and derived features 
* Importance ranking.

Table : Top 15 CatBoost Features

| Feature                                         | Importance |
|-------------------------------------------------|------------|
| DRUG_NAME_target_enc                            | 32.174900  |
| mutational_burden                               | 11.123910  |
| ploidy_wes                                      | 10.079438  |
| ploidy_snp6                                     | 8.811091   |
| Growth Properties_Suspension                    | 7.087490   |
| TARGET_PATHWAY_ERK MAPK signaling               | 2.179812   |
| TARGET_PATHWAY_DNA replication                  | 1.480679   |
| Growth Properties_Adherent                      | 1.030101   |
| TARGET_PATHWAY_PI3K/MTOR signaling              | 1.022666   |
| TARGET_PATHWAY_Apoptosis regulation             | 0.960653   |
| TCGA_DESC_NB                                    | 0.927613   |
| TARGET_PATHWAY_Unclassified                     | 0.795082   |
| GDSC Tissue descriptor 1_leukemia               | 0.785364   |
| GDSC Tissue descriptor 1_lung_NSCLC             | 0.691926   |
| TARGET_PATHWAY_EGFR signaling                   | 0.634410   |

Table : Top 15 XGBoost Features

| Feature                                         | Importance |
|-------------------------------------------------|------------|
| Growth Properties_Suspension                    | 0.076342   |
| TARGET_PATHWAY_Mitosis                          | 0.071803   |
| DRUG_NAME_target_enc                            | 0.048909   |
| TARGET_PATHWAY_Protein stability and degradation| 0.026582   |
| GDSC Tissue descriptor 1_leukemia               | 0.020315   |
| TARGET_PATHWAY_Cell cycle                       | 0.017737   |
| GDSC Tissue descriptor 2_pancreas               | 0.017495   |
| GDSC Tissue descriptor 2_melanoma               | 0.014538   |
| GDSC Tissue descriptor 2_testis                 | 0.012309   |
| GDSC Tissue descriptor 2_adrenal_gland          | 0.012272   |
| TCGA_DESC_NB                                    | 0.012262   |
| TCGA_DESC_SCLC                                  | 0.011314   |
| GDSC Tissue descriptor 1_lymphoma               | 0.010958   |
| GDSC Tissue descriptor 1_lung_SCLC              | 0.010861   |
| GDSC Tissue descriptor 2_digestive_system_other | 0.010577   |


## Algorithm
* Description or images of data flow graph
  * if AzureML, link to:
    * Training experiment
    * Scoring workflow
* What learner(s) were used?
* Learner hyper-parameters

Table : Model Hyperparameters used

| Parameter           | CatBoost   | XGBoost          |
|---------------------|------------|------------------|
| bagging_temperature | 0.43032    | NaN              |
| colsample_bytree    | NaN        | 0.70303          |
| depth               | 12         | NaN              |
| eval_metric         | R2         | NaN              |
| gamma               | NaN        | 0.112268         |
| iterations          | 2100       | NaN              |
| l2_leaf_reg         | 4.18484    | NaN              |
| learning_rate       | 0.127225   | 0.048373         |
| loss_function       | RMSE       | NaN              |
| max_depth           | NaN        | 9                |
| min_child_weight    | NaN        | 4                |
| min_data_in_leaf    | 22         | NaN              |
| n_estimators        | NaN        | 2000             |
| n_jobs              | NaN        | -1               |
| objective           | NaN        | reg:squarederror |
| random_state        | 42         | 42               |
| random_strength     | 0.229978   | NaN              |
| reg_alpha           | NaN        | 0.000255         |
| reg_lambda          | NaN        | 3.099565         |
| subsample           | 0.816038   | 0.918038         |
| tree_method         | NaN        | hist             |
| verbose             | 0          | NaN              |


## Results
* ROC/Lift charts, AUC, R^2, MAPE as appropriate
* Performance graphs for parameters sweeps if applicable

Table : The Model Performance for the Hybrid-Approach
| Model                 |   RMSE   |   MAE    |    R²     |
|-----------------------|----------|----------|-----------|
| Stacked linear layer  | 1.032433 | 0.769603 | 0.860284  |
| Average blend         | 1.032626 | 0.769966 | 0.860231  |
| CatBoost only         | 1.036871 | 0.772769 | 0.859080  |
| XGBoost only          | 1.041301 | 0.776740 | 0.857873  |

