=============
autoCV Module
=============

Description : 
 - This module is used for model selection:
    * Automate the models training with cross validation
    * GridSearch the best parameters
    * Export the optimized models as pkl files, and saved them in /pkl folders
    * Validate the optimized models, and select the best model  

- Class
    * dynaClassifier : Focus on classification problems
        -  fit() : fit method for classifier
    * dynaRegressor : Focus on regression problems
        -  fit() : fit method for regressor

 - Current available estimators
    * clf_cv : Class focusing on classification estimators
        - lgr - Logistic Regression (aka logit, MaxEnt) classifier - LogisticRegression()
        - svm : C-Support Vector Classification - SVM.SVC()
        - mlp : Multi-layer Perceptron classifier - MLPClassifier()
        - ada : An AdaBoost classifier - AdaBoostClassifier()
        - rf : Random Forest classifier - RandomForestClassifier()
        - gb : Gradient Boost classifier - GradientBoostingClassifier()
        - xgb : XGBoost classifier - xgb.XGBClassifier()
        - lsvc : Linear Support Vector Machine Classifier - LinearSVC()
        - hgboost : Hist Gradient Boosting classifier - HistGradientBoostingClassifier()
        - sgd : SDG classifier - SGDClassifier() 
        - rgcv : Ridge Cross-validation classifier - RidgeClassifierCV() 
    * reg_cv : Class focusing on regression estimators
        - lr : Linear Regression - LinearRegression()
        - knn : Regression based on k-nearest neighbors - KNeighborsRegressor()
        - svr : Epsilon-Support Vector Regression - SVM.SVR()
        - rf : Random Forest Regression - RandomForestRegressor()
        - ada : An AdaBoost regressor - AdaBoostRegressor()
        - gb : Gradient Boosting for regression -GradientBoostingRegressor()
        - tree : A decision tree regressor - DecisionTreeRegressor()
        - mlp : Multi-layer Perceptron regressor - MLPRegressor()
        - xgb : XGBoost regression - XGBRegressor()
        - hgboost : Hist Gradient Boosting regression - HistGradientBoostingRegressor()
        - huber : Huber regression - HuberRegressor()
        - rgcv : Ridge cross validation regression - RidgeCV()
        - cvlasso : Lasso cross validation regression - LassoCV()
        - sgd : Stochastic Gradient Descent regression - SGDRegressor()

dynaClassifier
---------------------

.. autoclass:: optimalflow.autoCV.dynaClassifier
    :members:

dynaRegressor
---------------------

.. autoclass:: optimalflow.autoCV.dynaRegressor
    :members:

evaluate_model
---------------------

.. autoclass:: optimalflow.autoCV.evaluate_model
    :members:

clf_cv
---------------------

.. autoclass:: optimalflow.estimatorCV.clf_cv

reg_cv
---------------------

.. autoclass:: optimalflow.estimatorCV.reg_cv

data_splitting_tool
---------------------

.. autofunction:: optimalflow.utilis_func.data_splitting_tool

reset_parameters
---------------------

.. autofunction:: optimalflow.utilis_func.reset_parameters

update_parameters
---------------------

.. autofunction:: optimalflow.utilis_func.update_parameters


export_parameters
---------------------

.. autofunction:: optimalflow.utilis_func.export_parameters


Defaults Parameters for Classifiers/Regressors
----------------------------------------------

**Estimators default parameters setting:**

.. list-table:: Classifiers Estimators Default Parameters Searching Range
   :widths: 25 25 50
   :header-rows: 1

   * - Estimators:
     - Parameters:
     - Value Range:
   * - lgr
     - 'C'
     - [0.001, 0.01, 0.1, 1, 10, 100, 1000]
   * - svm
     - 'C'
     - [0.1, 1, 10]
   * - 
     - 'kernel'
     - ['linear', 'poly', 'rbf', 'sigmoid']
   * - mlp
     - 'activation'
     - ['identity','relu', 'logistic']
   * - 
     - 'hidden_layer_sizes'
     - [10, 50, 100]
   * - 
     - 'learning_rate'
     - ['constant', 'invscaling', 'adaptive']
   * - 
     - 'solver'
     - ['lbfgs', 'sgd', 'adam']
   * - ada
     - 'n_estimators'
     - [50,100,150]
   * - 
     - 'learning_rate'
     - [0.01,0.1, 1, 5, 10]
   * - rf
     - 'max_depth'
     - [2, 4, 8, 16, 32]
   * - 
     - 'n_estimators'
     - [5, 50, 250]
   * - gb
     - 'n_estimators'
     - [50,100,150,200,250,300]
   * - 
     - 'max_depth'
     - [1, 3, 5, 7, 9]
   * - 
     - 'learning_rate' 
     - [0.01, 0.1, 1, 10, 100]
   * - xgb
     - 'n_estimators'
     - [50,100,150,200,250,300]
   * - 
     - 'max_depth'
     - [3, 5, 7, 9]
   * - 
     - 'learning_rate' 
     - [0.01, 0.1, 0.2,0.3,0.4]
   * - lsvc
     - 'C'
     - [0.1, 1, 10]
   * - sgd
     - 'penalty'
     - ['l2', 'l1', 'elasticnet']
   * - hgboost
     - 'max_depth' 
     - [3, 5, 7, 9]
   * - 
     - 'learning_rate' 
     - [0.1, 0.2,0.3,0.4]
   * - rgcv
     - 'fit_intercept' 
     - [True,False]
..

.. list-table:: Regressors Default Parameters Searching Range
   :widths: 25 25 50
   :header-rows: 1

   * - Estimators:
     - Parameters:
     - Value Range:
   * - lr
     - 'normalize'
     - [True,False]
   * - svm
     - 'C'
     - [0.1, 1, 10]
   * - 
     - 'kernel'
     - ['linear', 'poly', 'rbf', 'sigmoid']
   * - mlp
     - 'activation'
     - ['identity','relu', 'tanh', 'logistic']
   * - 
     - 'hidden_layer_sizes'
     - [10, 50, 100]
   * - 
     - 'learning_rate'
     - ['constant', 'invscaling', 'adaptive']
   * - 
     - 'solver'
     - ['lbfgs',  'adam']
   * - ada
     - 'n_estimators'
     - [50,100,150,200,250,300]
   * - 
     - 'loss'
     - ['linear','square','exponential']
   * - 
     - 'learning_rate'
     - [0.01, 0.1, 0.2,0.3,0.4]
   * - tree
     - 'splitter'
     - ['best', 'random']
   * - 
     - 'max_depth' 
     - [1, 3, 5, 7, 9]
   * - 
     - 'min_samples_leaf'
     - [1,3,5]
   * - rf
     - 'max_depth'
     - [2, 4, 8, 16, 32]
   * - 
     - 'n_estimators'
     - [5, 50, 250]
   * - gb
     - 'n_estimators'
     - [50,100,150,200,250,300]
   * - 
     - 'max_depth'
     - [3, 5, 7, 9]
   * - 
     - 'learning_rate' 
     - [0.01, 0.1, 0.2,0.3,0.4]
   * - xgb
     - 'n_estimators'
     - [50,100,150,200,250,300]
   * - 
     - 'max_depth'
     - [3, 5, 7, 9]
   * - 
     - 'learning_rate' 
     - [0.01, 0.1, 0.2,0.3,0.4] 
   * - sgd
     - 'shuffle'
     - [True,False]
   * - 
     - 'penalty'
     - ['l2', 'l1', 'elasticnet']
   * - 
     - 'learning_rate' 
     - ['constant','optimal','invscaling'] 
   * - cvlasso
     - 'fit_intercept'
     - [True,False]
   * - rgcv
     - 'fit_intercept'
     - [True,False]
   * - huber
     - 'fit_intercept'
     - [True,False]
   * - hgboost
     - 'max_depth'
     - [3, 5, 7, 9]
   * - 
     - 'learning_rate'
     - [0.1, 0.2,0.3,0.4]
..
