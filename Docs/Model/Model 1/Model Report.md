# Model Report
_A report to provide details on a specific experiment (model) - possibly one of many_

If applicable, the Automated Modeling and Reporting utility developed by Microsoft TDSP team can be used to generate reports, which can provide contents for most of the sections in this model report. 
## Analytic Approach
* What is target definition
* What are inputs (description)
* What kind of model was built?

## Model Description

* Models and Parameters

	* Description or images of data flow graph
  		* if AzureML, link to:
    		* Training experiment
    		* Scoring workflow
	* What learner(s) were used?
	* Learner hyper-parameters


## Results (Model Performance)
* ROC/Lift charts, AUC, R^2, MAPE as appropriate
* Performance graphs for parameters sweeps if applicable


  
| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| CatBoost tuned | 1.041411 | 0.777037 | 0.857843 |
| XGBoost tuned | 1.041933 | 0.777403 | 0.857700 |
| LightGBM tuned | 1.047572 | 0.781412 | 0.856156 |


## Model Understanding

* Variable Importance (significance)

* Insight Derived from the Model

Figure 2 : Top 15 CatBoost tuned features
### Top 15 CatBoost tuned features

| Feature | Importance |
|---|---:|
| DRUG_NAME_target_enc | 31.630180 |
| mutational_burden | 10.468298 |
| ploidy_wes | 10.291938 |
| ploidy_snp6 | 8.620349 |
| Growth Properties_Suspension | 6.862064 |
| TARGET_PATHWAY_ERK MAPK signaling | 2.362935 |
| TARGET_PATHWAY_DNA replication | 1.287688 |
| TARGET_PATHWAY_Apoptosis regulation | 1.079955 |
| TCGA_DESC_SCLC | 1.076396 |
| TARGET_PATHWAY_PI3K/MTOR signaling | 0.981120 |
| GDSC Tissue descriptor 1_leukemia | 0.896726 |
| TCGA_DESC_NB | 0.890713 |
| Growth Properties_Adherent | 0.806110 |
| TARGET_PATHWAY_EGFR signaling | 0.747406 |
| GDSC Tissue descriptor 2_melanoma | 0.721836 |

Figure 3 : Top 15 XGBoost tuned features
### Top 15 XGBoost tuned features

| Feature | Importance |
|---|---:|
| TARGET_PATHWAY_Mitosis | 0.085053 |
| Growth Properties_Suspension | 0.066671 |
| DRUG_NAME_target_enc | 0.045802 |
| TARGET_PATHWAY_Protein stability and degradation | 0.029176 |
| GDSC Tissue descriptor 1_leukemia | 0.021940 |
| TARGET_PATHWAY_Cell cycle | 0.018323 |
| GDSC Tissue descriptor 2_digestive_system_other | 0.012969 |
| TARGET_PATHWAY_JNK and p38 signaling | 0.011518 |
| TARGET_PATHWAY_Metabolism | 0.011395 |
| GDSC Tissue descriptor 2_melanoma | 0.011258 |
| TCGA_DESC_SCLC | 0.011245 |
| GDSC Tissue descriptor 1_blood | 0.011058 |
| GDSC Tissue descriptor 2_adrenal_gland | 0.010900 |
| GDSC Tissue descriptor 1_lymphoma | 0.010773 |
| GDSC Tissue descriptor 2_pancreas | 0.010768 |

Figure 4 : Top 15 LightGBM tuned features
### Top 15 LightGBM tuned features

| Feature | Importance |
|---|---:|
| DRUG_NAME_target_enc | 71568 |
| mutational_burden | 50584 |
| ploidy_snp6 | 47880 |
| ploidy_wes | 44492 |
| TARGET_PATHWAY_DNA replication | 4428 |
| TARGET_PATHWAY_PI3K/MTOR signaling | 4253 |
| TARGET_PATHWAY_ERK MAPK signaling | 3831 |
| TARGET_PATHWAY_Unclassified | 3402 |
| TARGET_PATHWAY_Other, kinases | 2588 |
| TARGET_PATHWAY_Apoptosis regulation | 2567 |
| Growth Properties_Adherent | 2448 |
| TARGET_PATHWAY_EGFR signaling | 2288 |
| TARGET_PATHWAY_Chromatin other | 2202 |
| TARGET_PATHWAY_Mitosis | 2049 |
| TARGET_PATHWAY_RTK signaling | 2047 |

## Conclusion and Discussions for Next Steps

* Conclusion

* Discussion on overfitting (if applicable)

* What other Features Can Be Generated from the Current Data

* What other Relevant Data Sources Are Available to Help the Modeling
