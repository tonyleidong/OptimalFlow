=============
autoFS Module
=============

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

dynaFS_clf
---------------------

.. autoclass:: dynapipe.autoFS.dynaFS_clf
    :members:

dynaFS_reg
---------------------

.. autoclass:: dynapipe.autoFS.dynaFS_reg
    :members:

clf_fs
---------------------

.. autoclass:: dynapipe.selectorFS.clf_fs

reg_fs
---------------------

.. autoclass:: dynapipe.selectorFS.reg_fs