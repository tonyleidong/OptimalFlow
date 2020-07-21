============
Modules
============

1. autoFS - Features Selection Module
===============
Description : 
 - This module is used for features selection:
    * Automate the feature selection with several selectors
    * Evaluate the outputs from all selector methods, and ranked a final list of the top important features
 
 - Class:
    * dynaFS_clf : Focus on classification problems
        - fit_fs_clf() - fit method for classifier
    * dynaFS_reg : Focus on regression problems
        - fit_fs_reg() - fit method for regressor

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


2. autoCV - Model selection module
===============
Description : 
 - This module is used for model selection:
    * Automate the models training with cross validation
    * GridSearch the best parameters
    * Export the optimized models as pkl files, and saved them in /pkl folders
    * Validate the optimized models, and select the best model 

- Class
    * dynaClassifier : Focus on classification problems
        -  fit_clf() : fit method for classifier
    * dynaRegressor : Focus on regression problems
        -  fit_reg() : fit method for regressor

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
