============
Modules
============
1. autoPP - Feature Preprocessing Module
===============

.. autoclass:: dynapipe.autoPP.dynaPreprocessing
    :members:

Description : 
 - This module is used for data preprocessing operation:
    * Impute with missing value
    * Winsorize with outlier
    * Scaling using popular scaler approaches
    * Encoding category features using popular encoder approaches
    * Generated all combination datasets for further modeling and evaluation
    * Sparsity calculation as the critera for output datasets filtering
    * Custom parameters initial settings, add/remove winsorization, scaling, or encoding strategies.
 
 - Class:
    * dynaPreprocessing : Focus on classification/regression prprocessing problems
        - fit() - fit & transform method for preprocessing

 - Current available strategies:
    * Scaling : Numeric features scaling, default settings
        - "standard" : StandardScaler() approach
        - "minmax" - MinMaxScaler() approach
        - "maxabs" - MaxAbsScaler() approach
        - "robust" - RobustScaler() approach
        
    * Encoding : Category features encoding, default settings
        - "onehot" : OnehotEncoder() approach, with dummy trap consideration in regression problem
        - "label" : LabelEncoder() approach
        - "frequency" : Frequency calculation approach  
        - "mean" : Mean calculation approach

    * winsorization : Default limits settings
        - (0.01,0.01) : Top 1% and bottom 1% will be excluded
        - (0.05,0.05) : Top 5% and bottom 5% will be excluded


        
2. autoFS - Features Selection Module
===============
Description : 
 - This module is used for features selection:
    * Automate the feature selection with several selectors
    * Evaluate the outputs from all selector methods, and ranked a final list of the top important features
 
 - Class:
    * dynaFS_clf : Focus on classification problems
        - fit() - fit and transform method for classifier
    * dynaFS_reg : Focus on regression problems
        - fit() - fit and transform method for regressor

 - Current available selectors
    * clf_fs : Class focusing on classification features selection
        - kbest_f : SelectKBest() with f_classif core
        - kbest_chi2 - SelectKBest() with chi2 core
        - rfe_lr - RFE with LogisticRegression() estimator
        - rfe_svm - RFE with SVC() estimator
        - rfecv_svm - RFECV with SVC() estimator  
        - rfe_tree - RFE with DecisionTreeClassifier() estimator
        - rfecv_tree - RFECV with DecisionTreeClassifier() estimator
        - rfe_rf - RFE with RandomForestClassifier() estimator
        - rfecv_rf - RFECV with RandomForestClassifier() estimator
        
    * reg_fs : Class focusing on regression features selection
        - kbest_f : SelectKBest() with f_regression core
        - rfe_svm : RFE with SVC() estimator
        - rfecv_svm : RFECV with SVC() estimator  
        - rfe_tree : RFE with DecisionTreeRegressor() estimator
        - rfecv_tree : RFECV with DecisionTreeRegressor() estimator
        - rfe_rf : RFE with RandomForestRegressor() estimator
        - rfecv_rf : RFECV with RandomForestRegressor() estimator


3. autoCV - Model selection module
===============
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

.. autoclass:: dynapipe.autoFS.dynaClassifier
    :members:
    
.. autoclass:: dynapipe.autoFS.dynaRegressor
    :members:

***Estimators default parameters setting:**

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
     - ['identity','relu', 'tanh', 'logistic']
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
..

4. autoPipe - Pipeline Module for Connection
===============
Description : 
 - This module is used for pipeline connection:
    * Connect autoPP, autoFS, and autoCV like scikit-learn pipeline approach.
    * Automated the iteration of cross-experiment to find the best baseline model
    * Generate comparable and parameter-tracable dictionaies and reports to support autoVIZ and autoFlow modules
 
 - Pipeline Components:
    * autoPP - dynaPreprocessing() Class in autoPP module
    * Datasets Splitting - pipeline_splitting_rule() Function in utilis_funs module
    * autoFS - dynaFS_clf() or dynaFS_reg() Class in autoFS module
    * autoCV - dynaClassifier() or dynaRegressor() Class in autoCV module
    * Model Evaluate - evaluate_model() Class in autoCV module
 
