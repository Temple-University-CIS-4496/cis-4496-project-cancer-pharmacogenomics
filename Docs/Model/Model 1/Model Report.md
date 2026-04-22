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



## Conclusion and Discussions for Next Steps

* Conclusion

* Discussion on overfitting (if applicable)

* What other Features Can Be Generated from the Current Data

* What other Relevant Data Sources Are Available to Help the Modeling
