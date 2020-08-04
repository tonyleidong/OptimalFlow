## Dynamic Pipeline
[![PyPI Latest Release](https://img.shields.io/pypi/v/dynapipe)](https://pypi.org/project/dynapipe/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/dynapipe?style=plastic)](https://pypi.org/project/dynapipe/)
[![Github Issues](https://img.shields.io/github/issues/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/DynamicPipeline/issues)
[![License](https://img.shields.io/github/license/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/DynamicPipeline/blob/master/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/dynapipe)
[![Python Version](https://img.shields.io/pypi/pyversions/dynapipe)](https://pypi.org/project/dynapipe/)


   
#### Author: [Tony Dong](http://www.linkedin.com/in/lei-tony-dong)

<img src="https://github.com/tonyleidong/DynamicPipeline/blob/master/docs/DynamicPipeline_Logo.png" width="80">**Dynamic Pipeline** is a high-level API toolkit to help data scientists building models in ensemble way, and automating Machine Learning workflow with simple codes. 

Comparing other popular "AutoML or Automatic Machine Learning" APIs, **Dynamic Pipeline** is designed as an omni-ensembled ML workflow optimizer with higher-level API targeting to avoid manual repetitive train-along-evaluate experiments in general pipeline building. 

To achieve that, **Dynamic Pipeline** applies *Pipeline Cluster Traversal Experiments* algorithm to assemble all cross-matching pipelines covering major tasks of Machine Learning workflow, and apply traversal-experiment to search the optimal baseline model.

Besides, by modularizing all key pipeline components in reuseable packages, it allows all components to be custom tunable along with high scalability.


<img src="https://github.com/tonyleidong/DynamicPipeline/blob/master/docs/DynamicPipeline_Workflow.PNG" width="980">

The core concept in **Dynamic Pipeline** is *Pipeline Cluster Traversal Experiments*, which is a theory, first raised by Tony Dong during Genpact 2020 GVector Conference, to optimize and automate Machine Learning Workflow using ensemble pipelines algorithm.

Comparing other automatic or classic machine learning workflow's repetitive experiments using single pipeline, *Pipeline Cluster Traversal Experiments* is more powerful, with larger coverage scope, to find the best model without manual intervention, and also more flexible with elasticity to cope with unseen data due to its ensemble designs in each component.

<img src="https://github.com/tonyleidong/DynamicPipeline/blob/master/docs/PipelineClusterTraversalExperiments.PNG" width="980">


In summary, **Dynamic Pipeline** shares a few useful properties for data scientists:

* *Easy & less coding* - High-level APIs to implement *Pipeline Cluster Traversal Experiments*, and each ML component is highly automated and modularized;

* *Well ensembled* - Each key component is ensemble of popular algorithms w/ optimal hyperparameters tuning included;
      
* *Hardly omission* - Using *Pipeline Cluster Traversal Experiments*, to cross-experiment with combined permutated input datasets, feature selection, and model selection;
      
* *Scalable* - Each module could add new algorithms easily due to its ensemble and reuseable coding design;

* *Adaptable* - *Pipeline Cluster Traversal Experiments* makes it easier to adapt unseen datasets with the right pipeline;
      
* *Custom Modify Welcomed* - Support custom settings to add/remove algorithms or modify hyperparameters for elastic requirements.

### Documentation:  https://dynamic-pipeline.readthedocs.io/

### Installation
```python
pip install dynapipe
```

### Core Modules: 
 - autoPP for feature preprocessing
 - autoFS for classification/regression features selection
 - autoCV for classification/regression model selection and evaluation
 - autoPipe for *Pipeline Cluster Traversal Experiments* 
 
### Modules
#### autoFS - features selection module

###### Description : 
 - This module is used for features selection:
    * Automate the feature selection with several selectors
    * Evaluate the outputs from all selector methods, and ranked a final list of the top important features
    
###### Current methods & selectors covered:
```python
 - Class
    * dynaFS_clf - Focus on classification problems
        - $ fit_fs_clf() - fit method for classifier
    * dynaFS_reg - Focus on regression problems
        - $ fit_fs_reg() - fit method for regressor
 - Class and current available selectors
    * clf_fs - Class focusing on classification features selection
        $ kbest_f - SelectKBest() with f_classif core
        $ kbest_chi2 - SelectKBest() with chi2 core
        $ rfe_lr - RFE with LogisticRegression() estimator
        $ rfe_svm - RFE with SVC() estimator
        $ rfecv_svm - RFECV with SVC() estimator  
        $ rfe_tree - RFE with DecisionTreeClassifier() estimator
        $ rfecv_tree - RFECV with DecisionTreeClassifier() estimator
        $ rfe_rf - RFE with RandomForestClassifier() estimator
        $ rfecv_rf - RFECV with RandomForestClassifier() estimator
        
    * reg_fs - Class focusing on regression features selection
        $ kbest_f - SelectKBest() with f_regression core
        $ rfe_svm - RFE with SVC() estimator
        $ rfecv_svm - RFECV with SVC() estimator  
        $ rfe_tree - RFE with DecisionTreeRegressor() estimator
        $ rfecv_tree - RFECV with DecisionTreeRegressor() estimator
        $ rfe_rf - RFE with RandomForestRegressor() estimator
        $ rfecv_rf - RFECV with RandomForestRegressor() estimator
```

###### Demo - Regression Problem
- CODE:
```python
import pandas as pd
from dynapipe.autoFS import dynaFS_reg

tr_features = pd.read_csv('./data/regression/train_features.csv')
tr_labels = pd.read_csv('./data/regression/train_labels.csv')

reg_fs_demo = dynaFS_reg( fs_num = 5,random_state = 13,cv = 5,input_from_file = True)
reg_fs_demo.fit_fs_reg(tr_features,tr_labels,detail_info = False)
```
- OUTPUT:
```javascript
      *DynaPipe* autoFS Module ===> Selector kbest_f gets outputs: ['INDUS', 'NOX', 'RM', 'PTRATIO', 'LSTAT']
Progress: [###-----------------] 14.3%

      *DynaPipe* autoFS Module ===> Selector rfe_svm gets outputs: ['CHAS', 'NOX', 'RM', 'PTRATIO', 'LSTAT']
Progress: [######--------------] 28.6%

      *DynaPipe* autoFS Module ===> Selector rfe_tree gets outputs: ['CRIM', 'RM', 'DIS', 'TAX', 'LSTAT']
Progress: [#########-----------] 42.9%

      *DynaPipe* autoFS Module ===> Selector rfe_rf gets outputs: ['CRIM', 'RM', 'DIS', 'PTRATIO', 'LSTAT']
Progress: [###########---------] 57.1%

      *DynaPipe* autoFS Module ===> Selector rfecv_svm gets outputs: ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
Progress: [##############------] 71.4%

      *DynaPipe* autoFS Module ===> Selector rfecv_tree gets outputs: ['CRIM', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
Progress: [#################---] 85.7%

      *DynaPipe* autoFS Module ===> Selector rfecv_rf gets outputs: ['CRIM', 'ZN', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
Progress: [####################] 100.0%

The DynaPipe autoFS identify the top 5 important features for regression are: ['RM', 'LSTAT', 'PTRATIO', 'NOX', 'CRIM']. 
```


#### autoCV - model selection module
Description : 
 - This module is used for model selection:
    * Automate the models training with cross validation
    * GridSearch the best parameters
    * Export the optimized models as pkl, and saved in /pkl folders
    * Validate the optimized models, and select the best model 

###### Current methods & estimators covered:
```python
 - Class
    * dynaClassifier - Focus on classification problems
        - $ fit() - fit method for classifier
    * dynaRegressor - Focus on regression problems
        - $ fit() - fit method for regressor

 - Class and current available estimators
    * clf_cv - Class focusing on classification estimators
        $ lgr - Logistic Regression (aka logit, MaxEnt) classifier - LogisticRegression()
        $ svm - C-Support Vector Classification - SVM.SVC()
        $ mlp - Multi-layer Perceptron classifier - MLPClassifier()
        $ ada - An AdaBoost classifier - AdaBoostClassifier()
        $ rf - Random Forest classifier - RandomForestClassifier()
        $ gb - Gradient Boost classifier - GradientBoostingClassifier()
        $ xgb = XGBoost classifier - xgb.XGBClassifier()
    * reg_cv - Class focusing on regression estimators
        $ lr - Linear Regression - LinearRegression()
        $ knn - Regression based on k-nearest neighbors - KNeighborsRegressor()
        $ svr - Epsilon-Support Vector Regression - SVM.SVR()
        $ rf - Random Forest Regression - RandomForestRegressor()
        $ ada - An AdaBoost regressor - AdaBoostRegressor()
        $ gb - Gradient Boosting for regression -GradientBoostingRegressor()
        $ tree - A decision tree regressor - DecisionTreeRegressor()
        $ mlp - Multi-layer Perceptron regressor - MLPRegressor()
        $ xgb - XGBoost regression - XGBRegressor()
```
###### Demo - Classification Problem

- CODE:
```python
import pandas as pd
from dynapipe.autoCV import evaluate_model,dynaClassifier
import joblib

tr_features = pd.read_csv('./data/classification/train_features.csv')
tr_labels = pd.read_csv('./data/classification/train_labels.csv')
val_features = pd.read_csv('./data/classification/val_features.csv')
val_labels = pd.read_csv('./data/classification/val_labels.csv')

clf_cv_demo = dynaClassifier(random_state = 13,cv_num = 5,input_from_file = True)
clf_cv_demo.fit(tr_features,tr_labels,detail_info = False)

models = {}

for mdl in ['lr','knn','tree','svm','mlp','rf','gb','ada','xgb']:
    models[mdl] = joblib.load('./pkl/{}_cls_model.pkl'.format(mdl))

for name, mdl in models.items():
    try:
        ml_evl = evaluate_model(model_type = "cls")
        ml_evl.fit(name, mdl, val_features, val_labels)
    except:
        print(f"Failed to load the {mdl}.")
```
- OUTPUT:
```python


      *DynaPipe* autoMS Module ===> lgr_CrossValidation with 5 folds:

Best Parameters: {'C': 1, 'random_state': 13}

Best CV Score: 0.7997178628107917

Progress: [###-----------------] 14.3%

      *DynaPipe* autoCV Module ===> svm_CrossValidation with 5 folds:

Best Parameters: {'C': 0.1, 'kernel': 'linear'}

Best CV Score: 0.7959619114794568

Progress: [######--------------] 28.6%

      *DynaPipe* autoCV Module ===> mlp_CrossValidation with 5 folds:

Best Parameters: {'activation': 'tanh', 'hidden_layer_sizes': (50,), 'learning_rate': 'constant', 'random_state': 13, 'solver': 'lbfgs'}

Best CV Score: 0.8184094515958386

Progress: [#########-----------] 42.9%

      *DynaPipe* autoCV Module ===> rf_CrossValidation with 5 folds:

Best Parameters: {'max_depth': 4, 'n_estimators': 250, 'random_state': 13}

Best CV Score: 0.8240521953800035

Progress: [###########---------] 57.1%

      *DynaPipe* autoCV Module ===> ada_CrossValidation with 5 folds:

Best Parameters: {'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 13}

Best CV Score: 0.824034561805678

Progress: [##############------] 71.4%

      *DynaPipe* autoCV Module ===> gb_CrossValidation with 5 folds:

Best Parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 300, 'random_state': 13}

Best CV Score: 0.8408746252865456

Progress: [#################---] 85.7%

      *DynaPipe* autoCV Module ===> xgb_CrossValidation with 5 folds:

Best Parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'verbosity': 0}

Best CV Score: 0.8464292011990832

Progress: [####################] 100.0%
lgr -- Accuracy: 0.775 / Precision: 0.712 / Recall: 0.646 / Latency: 0.0ms
svm -- Accuracy: 0.747 / Precision: 0.672 / Recall: 0.6 / Latency: 2.0ms
mlp -- Accuracy: 0.787 / Precision: 0.745 / Recall: 0.631 / Latency: 4.1ms
rf -- Accuracy: 0.809 / Precision: 0.83 / Recall: 0.6 / Latency: 37.0ms
ada -- Accuracy: 0.792 / Precision: 0.759 / Recall: 0.631 / Latency: 21.4ms
gb -- Accuracy: 0.815 / Precision: 0.796 / Recall: 0.662 / Latency: 2.0ms
xgb -- Accuracy: 0.815 / Precision: 0.786 / Recall: 0.677 / Latency: 5.0ms
```
#### autoPP - features preprocessing module

###### Description : 
 - This module is used for features preprocessing:
    * Apply ensemble imputing, winsorization,encoding, and scaling steps to cleaned dataset. 
    * Generate all possible datasets after cross-combiniations and columns permutations in ensemble algorithm approach
    * Support custom sparsity and column numbers ceriteria to narrow the number of combination datasets 
    
#### autoPipe - pipeline module for connection

###### Description : 
 - This module is used for pipeline connection:
    * Connect autoPP, autoFS, and autoCV like scikit-learn pipeline approach.
    * Automated the iteration of cross-experiment to find the best baseline model
    * Generate comparable and parameter-tracable dictionaies and reports to support autoVIZ and autoFlow modules

###### Demo - Classification Problem
- CODE:
```python
import pandas as pd
from dynapipe.autoPipe import autoPipe
from dynapipe.funcPP import PPtools
from dynapipe.autoPP import dynaPreprocessing
from dynapipe.autoFS import dynaFS_clf
from dynapipe.autoCV import evaluate_model,dynaClassifier

df = pd.read_csv('./data/preprocessing/breast_cancer.csv')

pipe = autoPipe(
[("autoPP",dynaPreprocessing(custom_parameters = None, label_col = 'diagnosis', model_type = "cls")),
("datasets_splitting",pipeline_splitting_rule(val_size = 0.2, test_size = 0.2, random_state = 13)),
("autoFS",dynaFS_clf(fs_num = 5, random_state=13, cv = 5, in_pipeline = True, input_from_file = False)),
("autoCV",dynaClassifier(random_state = 13,cv_num = 5,in_pipeline = True, input_from_file = False)),
("model_evaluate",evaluate_model(model_type = "cls"))])

dyna_report= pipe.fit(df)[4]
dyna_report.head(5)

```
- OUTPUT:
```python

	Dataset	   Model_Name    Best_Parameters	 Accuracy	Precision	Recall	Latency
1	Dataset_0	svm	[('C', 0.1), ('kernel', 'linear')]	  0.930 0.889 0.96 3.0
6	Dataset_0	xgb	[('learning_rate', 1), ('max_depth', 2), ('n_estimators', 50), ('random_state', 13)]	0.912	0.955	0.84	2.0
40	Dataset_5	gb	[('learning_rate', 1), ('max_depth', 2), ('n_estimators', 50), ('random_state', 13)]	0.895	0.913	0.84	2.0
31	Dataset_4	rf	[('max_depth', 2), ('n_estimators', 50), ('random_state', 13)]	0.877	0.821	0.92	12.0
51	Dataset_7	mlp	[('activation', 'relu'), ('hidden_layer_sizes', (10,)), ('learning_rate', 'constant'), ('random_state', 13), ('solver', 'sgd')]	0.772	0.875	0.56	4.0
```
### License:
MIT
