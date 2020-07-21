============
Usage
============

1. Features selection for regression problem:

  Demo Code:

.. code-block:: python

   import pandas as pd
   from dynapipe.autoFS import dynaFS_reg

   tr_features = pd.read_csv('./data/regression/train_features.csv')
   tr_labels = pd.read_csv('./data/regression/train_labels.csv')
   
   # Set input_form_file = False, when label values are array. Select 'True' from Pandas dataframe.
   reg_fs_demo = dynaFS_reg( fs_num = 5,random_state = 13,cv = 5,input_from_file = True)
   # Select detail_info = True, when you want to see the detail of the iteration
   reg_fs_demo.fit_fs_reg(tr_features,tr_labels,detail_info = False)

  Output:

.. code-block:: python

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

2. Model selection for classification problem:

  Demo Code:

.. code-block:: python

   import pandas as pd
   from dynapipe.autoCV import dynaClassifier,evaluate_clf_model
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

Output:

.. code-block:: python
      
    *DynaPipe* autoCV Module ===> lgr_CrossValidation with 5 folds:

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

3. Default parameters settings:

  Currently, there're 3 methods in *utilis_fun* module - *reset_parameters*, *update_parameters*, and *export_parameters*.

  - *update_parameters* method is used to modify the default parameter settings for models selection module (autoCV).

  i.e. When you want to modify the support vector machine clssifier, with new penalty "C" and "kernel" values, the code line below will achieve that.

.. code-block:: python

 update_parameters(mode = "cls", estimator_name = "svm", C=[0.1,0.2],kernel=["linear"])

  - *export_parameters* method can help you export the currnt default parameter settings as 2 csv files named "exported_cls_parameters.csv" and "exported_reg_parameters.csv". You can find them in the *./exported* folder of you current work dictionary.

.. code-block:: python

 export_parameters()

  - *reset_parameters* method can reset the default parameter settings to the package's original default settings. Just add this code line will work:

.. code-block:: python

 reset_parameters()
