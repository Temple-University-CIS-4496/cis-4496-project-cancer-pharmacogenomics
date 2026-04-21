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

| Model                 |   RMSE    |   MAE     |    R²     |
|-----------------------|-----------|-----------|-----------|
| CatBoost              | 1.131118  | 0.844668  | 0.832297  |
| XGBoost               | 1.145858  | 0.854324  | 0.827898  |
| MLPRegressor          | 1.189713  | 0.893602  | 0.814473  |
| HistGradientBoosting  | 1.225448  | 0.914313  | 0.803160  |
| LightGBM              | 1.226727  | 0.915228  | 0.802749  |
| ExtraTrees            | 1.249409  | 0.917899  | 0.795387  |
| Ridge                 | 1.344436  | 1.008067  | 0.763079  |

  
 Model 				     | RMSE		|  MAE		| R2 |
XGBoost(w/ Optuna)   	1.032373	0.768512   0.8603
LightGBM(w/ Optuna) 	1.035314  	0.772103   0.859503
Catboost(w/ Optuna) 	1.050988 	0.784145   0.855216
HistGradient(w/ Optuna) 1.099774	0.821224   0.841463
Catboost				1.110923    0.829804   0.838232
Extra Trees				1.2494		0.9181	   0.7954
Ridge(w/ Optuna)		1.341		1.006	   0.76426895
Linear Regression		1.3443		1.0079	   0.7631
HistGradient			1.396169	1.047938   0.741699
MLPRegressor			1.409077	1.059455   0.7369

## Model Understanding (Allan)

* Variable Importance (significance)
  - The high cardinality feature, which was separately encoded as **DRUG_NAME_target_enc** is considered to be the biggest predictor in the baseline feature analysis, showing that identity carried the most weight when it came to the prediction for **LN_IC50**.
  - The merged numerical features of **ploidy_wes**, **ploidy_snp6**, and **mutational burden** were among the most informative features for the baseline models.
  - Other Cancer-context based features like **TCGA_DESC**, **GDSC_Tissue_descriptor_1**, **GDSC_Tissue_descriptor_2**, and **TARGET_PATHWAY** contributed to the model by suporting the signals that account for the tissue, disease-specific, and pathway response patterns.
* Insight Derived from the Model



## Conclusion and Discussions for Next Steps (Julie)

* Conclusion on Feasibility Assessment of the Machine Learning Task

* Discussion on Overfitting (If Applicable)

* What other Features Can Be Generated from the Current Data

* What other Relevant Data Sources Are Available to Help the Modeling
