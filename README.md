## Dynamic Pipeline
[![PyPI Latest Release](https://img.shields.io/pypi/v/dynapipe)](https://pypi.org/project/dynapipe/)
[![Github Issues](https://img.shields.io/github/issues/tonyleidong/dynapipe)](https://github.com/tonyleidong/dynapipe/issues)
[![License](https://img.shields.io/github/license/tonyleidong/dynapipe)](https://github.com/tonyleidong/dynapipe/blob/master/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/tonyleidong/dynapipe)](https://github.com/tonyleidong/dynapipe)
[![Python Version](https://img.shields.io/pypi/pyversions/dynapipe)](https://pypi.org/project/dynapipe/)


   
#### Author: [Tony Dong](http://www.linkedin.com/in/lei-tony-dong)

<img src="https://github.com/tonyleidong/DynamicPipeline/blob/master/docs/DynamicPipeline_Official_Logo.png" width="80">**Dynamic Pipeline** is a high-level API to help data scientists building models in ensemble way, and automating Machine Learning workflow with simple code.

<img src=".png" width="1200">

Documentation:  https://dynamic-pipeline.readthedocs.io/

### Current available modules: 
 - autoPP for feature preprocessing
 - autoFS for classification/regression features selection
 - autoCV for classification/regression model selection and evaluation
 - autoPipe for modules automatic pipeline connection & generate model's performance reports
 
### Modules in development:
 - autoVIZ for pipeline visualization 
 - autoFlow for tracking and deployment
 - autoTM for text mining
 - Unsupervised models specific modules

### Installation
```python
pip install dynapipe
```

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

###### Demo - Input from csv file - Regression Problem
- CODE:
```python
import pandas as pd
from dynapipe.autoFS import dynaFS_reg

tr_features = pd.read_csv('./data/regression/train_features.csv')
tr_labels = pd.read_csv('./data/regression/train_labels.csv')

# Set input_form_file = False, when label values are array. Select 'True' from Pandas dataframe.
reg_fs_demo = dynaFS_reg( fs_num = 5,random_state = 13,cv = 5,input_from_file = True)
# Select detail_info = True, when you want to see the detail of the iteration
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
        - $ fit_clf() - fit method for classifier
    * dynaRegressor - Focus on regression problems
        - $ fit_reg() - fit method for regressor

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
###### Demo - Input from csv file - Classification Problem

- CODE:
```python
import pandas as pd
from dynapipe.autoMS import dynaClassifier,evaluate_clf_model
import joblib

tr_features = pd.read_csv('./data/classification/train_features.csv')
tr_labels = pd.read_csv('./data/classification/train_labels.csv')
val_features = pd.read_csv('./data/classification/val_features.csv')
val_labels = pd.read_csv('./data/classification/val_labels.csv')

# Set input_form_file = False, when label values are array. Select 'True' from Pandas dataframe.
clf_cv_demo = dynaClassifier(random_state = 13,cv_num = 5,input_from_file = True)
# Select detail_info = True, when you want to see the detail of the iteration
clf_cv_demo.fit_clf(tr_features,tr_labels,detail_info = False)

models = {}

for mdl in ['lgr','svm','mlp','rf','ada','gb','xgb']:
    models[mdl] = joblib.load('./pkl/{}_clf_model.pkl'.format(mdl))

for name, mdl in models.items():
    evaluate_clf_model(name, mdl, val_features, val_labels)
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
### License:
MIT