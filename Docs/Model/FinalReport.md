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

<img width="1147" height="618" alt="Screenshot 2026-04-23 094942" src="https://github.com/user-attachments/assets/32eca5bc-2edf-4949-a22d-78392bf8cbbf" />


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

** For the final hybrid model, the biologically meaningful feature importance comes from the CatBoost and XGBoost base learners, because the meta-model only combines their prediction outputs.

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

Table 1:- Baseline Model Performance Using Default Settings; Compares RMSE, MAE, and R² across initial baseline models before hyperparameter tuning.
| Model                 | RMSE      | MAE       | R²        |
|-----------------------|-----------|-----------|-----------|
| CatBoost              | 1.132169  | 0.843563  | 0.831986  |
| XGBoost               | 1.140523  | 0.851656  | 0.829497  |
| MLPRegressor          | 1.182636  | 0.888743  | 0.816673  |
| LightGBM              | 1.229644  | 0.916978  | 0.801810  |
| HistGradientBoosting  | 1.230592  | 0.916503  | 0.801504  |
| ExtraTrees            | 1.232253  | 0.911635  | 0.800968  |
| Ridge                 | 1.344522  | 1.007612  | 0.763048  |

Table 2:- Baseline Model Performance After Hyperparameter Tuning; Shows improved model performance after preprocessing and tuning, with XGBoost achieving the strongest results.
|Model 				     |   RMSE	|    MAE	|      R²  	 |
|------------------------|----------|-----------|------------|
| XGBoost(w/ Optuna)   	 | 1.032373	| 0.768512  |  0.8603	 |
| LightGBM(w/ Optuna) 	 | 1.035314 | 0.772103  |  0.859503  |
| Catboost(w/ Optuna) 	 | 1.050988 | 0.784145  |  0.855216  |
| HistGradient(w/ Optuna)| 1.099774 | 0.821224  |  0.841463  |
| Catboost		         | 1.110923 | 0.829804  |  0.838232  |
| Extra Trees			 | 1.2494	| 0.9181	|  0.7954    |
| Ridge(w/ Optuna)		 | 1.341	| 1.006		|  0.76426895|
| Linear Regression		 | 1.3443	| 1.0079	|  0.7631    |
| HistGradient			 | 1.396169	| 1.047938  |  0.741699  | 
| MLPRegressor			 | 1.409077	| 1.059455  |  0.7369    |

Figure 3:- Baseline Actual vs. Predicted LN_IC50 Across Models;  Baseline models generally follow the diagonal trend, showing that LN_IC50 prediction is feasible, but stronger boosting models such as CatBoost and XGBoost show tighter prediction patterns than weaker models like Ridge.

![Actual vs Predicted LN_IC50 for Baseline Models](actual%20vs%20predicted.png)

After preprocessing, multiple supervised regression models were tested to compare how well they predicted LN_IC50. Based on Table 1, CatBoost performed best among the default baseline models, with the highest R²and the lowest RMSE/MAE. XGBoost followed closely, while MLPRegressor also showed reasonable predictive performance. LightGBM, HistGradientBoosting, and ExtraTrees performed similarly but slightly lower, and Ridge had the weakest results, suggesting that a simple linear model was less effective for this dataset. The table shows the RMSE results for XGBoost at the lowest of about 1.003 which in comparison to the predicted values, is better post hypertuning. The predicted best model was supposed to be CatBoost, but post hyperparameter tuning and preprocessing, it was the third best model. LightGBM came in second for the best. XGBoost uses its fast-paced decision-based machine learning methods to distribute decision trees leading the machine learning techniques for regression, classification, and ranking problems. Due to the high volume of data, nonlinear tree-based and boosting models were better suited for capturing the relationships between drug, genomic, tissue, and pathway features and LN_IC50. Table 2 further shows the hyperparameter tuning improved performance with XGBoost using Optuna. Optuna was utilized to automate and accelerate machine learning techniques. As seen in Figure 3, there is a relatively even distribution across the trend line for all of the predicted vs actual LN_IC50 values. This emphasizes the improvements methods such as preprocessing and hyperparameter tuning to cause the even distribution. 



