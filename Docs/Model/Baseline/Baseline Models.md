# Baseline Model Report

_Baseline model is the the model a data scientist would train and evaluate quickly after he/she has the first (preliminary) feature set ready for the machine learning modeling. Through building the baseline model, the data scientist can have a quick assessment of the feasibility of the machine learning task._

When applicable, the Automated Modeling and Reporting utility developed by TDSP team of Microsoft is employed to build the baseline models quickly. The baseline model report is generated from this utility easily. 

> If using the Automated Modeling and Reporting tool, most of the sections below will be generated automatically from this tool. 

## Analytic Approach (hiu)
* What is target definition
* What are inputs (description)
* What kind of model was built?
  
The primary target variable for this project is LN_IC50, defined as the natural logarithm of the half-maximal inhibitory concentration. LN_IC50 was selected because it provides the best balance of statistical dispersion, biological interpretability, and alignment with established GDSC drug-response endpoints, compared to alternatives such as AUC and Z-scores. The modeling task is framed as a supervised regression problem, with LN_IC50 treated as a continuous outcome, enabling precise prediction of drug sensitivity across diverse cancer cell lines.

The model inputs were constructed from a carefully engineered set of biological, genomic, tissue, and drug-related features. The final modeling dataset includes drug identity (DRUG_NAME), mechanism-level information describing drug targets (TARGET_PATHWAY), standardized cancer classifications (TCGA_DESC), and tissue-level descriptors (GDSC Tissue descriptor 1 and GDSC Tissue descriptor 2). These are complemented by phenotypic indicators such as microsatellite instability (MSI) and growth properties, as well as genomic predictors including mutational_burden and ploidy estimates derived from SNP array and whole-exome sequencing (ploidy_snp6 and ploidy_wes). The feature engineering pipeline included imputing missing values, one-hot encoding low-cardinality categorical variables, target encoding high-cardinality features such as DRUG_NAME, and standardization of continuous predictors. In addition, feature selection was supported using mutual information analysis and repeated subsampling to retain the most informative predictors.

The analytic strategy initially evaluated a broad set of models, including linear, tree-based, and neural network approaches. Then, it focused on gradient boosting methods, such as CatBoost, XGBoost, and LightGBM, which were the most effective at capturing the nonlinear relationships and feature interactions inherent in drug response data. To improve methodological rigor, the pipeline incorporated cross-validation target encoding. This ensured that encoded features were generated out-of-fold, reducing data leakage and producing more reliable performance estimates. Although this stricter encoding initially resulted in slightly lower performance, subsequent hyperparameter tuning restored and improved model effectiveness.

The final modeling approach used a stacked hybrid architecture that combined a CatBoost and an XGBoost as base models with a second-layer linear regression model. This design leverages the complementary strengths of the models and improves predictive performance beyond that of the individual models. Model performance was evaluated using R^2, RMSE, and MAE, which provide a balanced assessment of explained variance, sensitivity to large errors, and typical prediction accuracy.



## Model Description (mo)

* Models and Parameters

	* Description or images of data flow graph
  		* if AzureML, link to:
    		* Training experiment
    		* Scoring workflow
	* What learner(s) were used?
	* Learner hyper-parameters


## Results (Model Performance) (Grace)
* ROC/Lift charts, AUC, R^2, MAPE as appropriate
* Performance graphs for parameters sweeps if applicable

Figure 1: Chart of all tested models default as a representation of good values. 
| Model                 | RMSE      | MAE       | R²        |
|-----------------------|-----------|-----------|-----------|
| CatBoost              | 1.132169  | 0.843563  | 0.831986  |
| XGBoost               | 1.140523  | 0.851656  | 0.829497  |
| MLPRegressor          | 1.182636  | 0.888743  | 0.816673  |
| LightGBM              | 1.229644  | 0.916978  | 0.801810  |
| HistGradientBoosting  | 1.230592  | 0.916503  | 0.801504  |
| ExtraTrees            | 1.232253  | 0.911635  | 0.800968  |
| Ridge                 | 1.344522  | 1.007612  | 0.763048  |


Figure 2: Chart of the tested models using the dataset with the preprocessing and hyperparameter tuning. 
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

Figure 3: Heatmap of the different TCGA_DESC values and DRUG_NAME with the LN_IC50 values. The lower the IC50 the more sensitive it is to treating the cancer type. The higher the IC50 is the less sensitive/likely it is to treat the cancer type. 

<img width="1440" height="864" alt="IC50 HeatMap (All Cancers)" src="https://github.com/user-attachments/assets/94a79371-7ed7-449c-9512-6f5ed3ed86c0" />
As seen in the heatmap, Daporinad shows consistenly low IC50 across the board has a low IC50 indicating high sensitivity in vitro. This consistently likely reflects the effects of NAMPT (nicotinamide phosphoribosyltransferase) inhibitor, which targets a fundamental metabolic pathway that cancer relies on. 

![Actual vs Predicted LN_IC50 for Baseline Models](Docs/Model/Baseline/actual%20vs%20predicted.png)



## Model Understanding (Allan)

* Variable Importance (significance)
  
  - The high cardinality feature, which was separately encoded as **DRUG_NAME_target_enc** is considered to be the biggest predictor in the baseline feature analysis, showing that identity carried the most weight when it came to the prediction for **LN_IC50**.
  - The merged numerical features of **ploidy_wes**, **ploidy_snp6**, and **mutational burden** were among the most informative features for the baseline models.
  - Other Cancer-context based features like **TCGA_DESC**, **GDSC_Tissue_descriptor_1**, **GDSC_Tissue_descriptor_2**, and **TARGET_PATHWAY** contributed to the model by suporting the signals that account for the tissue, disease-specific, and pathway response patterns.
    
* Insight Derived from the Model
  
  - The baseline models suggest that drug response is mainly explained by a combination of drug identity and biological context, not by one feature alone. The strong importance of DRUG_NAME_target_enc shows that the specific drug being tested carries the largest share of predictive signal, which means different drugs already have distinct response patterns in the dataset.
  - The high ranking of ploidy_wes, ploidy_snp6, and mutational_burden suggests that genomic instability and mutation load help explain why some cell lines are more sensitive or resistant to treatment. The contribution of TCGA_DESC, GDSC Tissue descriptor 1, and GDSC Tissue descriptor 2 suggests that cancer type and tissue background provide important context for interpreting response, even if they are not as dominant as drug identity.
  - The fact that nonlinear models such as CatBoost and XGBoost performed best supports the idea that the relationship between predictors and LN_IC50 is likely complex and interaction-based, rather than purely linear.
  - Overall, the baseline results suggest that cancer drug sensitivity is influenced by interacting factors across drug, tumor, and genomic levels, which supports the decision to use more flexible machine learning models for later tuned and hybrid comparisons.

Figure 3: Top 15 Features of Catboost
| Feature                                      | Importance |
|----------------------------------------------|------------|
| DRUG_NAME_target_enc                         | 52.233626  |
| ploidy_wes                                   | 7.435730   |
| mutational_burden                            | 6.868373   |
| Growth_Properties_Suspension                 | 6.782117   |
| ploidy_snp6                                  | 6.470171   |
| TARGET_PATHWAY_ERK_MAPK_signaling            | 1.713995   |
| GDSC_Tissue_descriptor_1_leukemia            | 1.342382   |
| GDSC_Tissue_descriptor_1_lymphoma            | 0.816106   |
| Growth_Properties_Adherent                   | 0.618551   |
| TCGA_DESC_NB                                 | 0.613951   |
| TCGA_DESC_SCLC                               | 0.569806   |
| TARGET_PATHWAY_DNA_replication               | 0.492163   |
| TARGET_PATHWAY_EGFR_signaling                | 0.424365   |
| GDSC_Tissue_descriptor_2_melanoma            | 0.404309   |
| TCGA_DESC_PAAD                               | 0.389557   |

Figure 4: Top 15 Features of XgBoost
| Feature                                              | Importance |
|------------------------------------------------------|------------|
| Growth_Properties_Suspension                         | 0.135220   |
| DRUG_NAME_target_enc                                 | 0.126386   |
| GDSC_Tissue_descriptor_1_leukemia                    | 0.022176   |
| TCGA_DESC_SCLC                                       | 0.021394   |
| GDSC_Tissue_descriptor_2_melanoma                    | 0.021156   |
| TCGA_DESC_PAAD                                       | 0.018818   |
| GDSC_Tissue_descriptor_1_blood                       | 0.017471   |
| GDSC_Tissue_descriptor_2_lymphoid_neoplasm_other     | 0.017121   |
| GDSC_Tissue_descriptor_1_lymphoma                    | 0.016868   |
| GDSC_Tissue_descriptor_2_rhabdomyosarcoma            | 0.016307   |
| GDSC_Tissue_descriptor_2_head_and_neck               | 0.015500   |
| GDSC_Tissue_descriptor_2_fibrosarcoma                | 0.013792   |
| TCGA_DESC_MB                                         | 0.013783   |
| TCGA_DESC_NB                                         | 0.012442   |
| GDSC_Tissue_descriptor_2_mesothelioma                | 0.012044   |


## Conclusion and Discussions for Next Steps (Julie)

* Conclusion on Feasibility Assessment of the Machine Learning Task

* Discussion on Overfitting (If Applicable)

* What other Features Can Be Generated from the Current Data

* What other Relevant Data Sources Are Available to Help the Modeling
