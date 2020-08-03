========
Examples
========

Feature preprocessing for a regression problem using autoPP:
------------------------------------------------------------

  Demo Code:

.. code-block:: python

  import pandas as pd 
  from funcPP import PPtools
  from autoPP import dynaPreprocessing

  df = pd.read_csv('../data/preprocessing/breast_cancer.csv')

  custom_parameters = {
      "scaler" : ["None", "standard"],
       # threshold number of category dimension
      "encode_band" : [6],
       # low dimension encoding
      "low_encode" : ["onehot","label"], 
       # high dimension encoding
      "high_encode" : ["frequency", "mean"],
      "winsorizer" : [(0.1,0.1)],
      "sparsity" : [0.4862],
      "cols" : [27]
     }
  dyna = dynaPreprocessing(custom_parameters = custom_parameters, label_col = 'diagnosis', model_type = "reg")
  dict_df = dyna.fit(input_data = df)
  print(f"Total combinations: {len(dict_df.keys())}")
  print(dict_df['winsor_0-Scaler_standard-Dataset_441'])

..

 Output:

.. code-block:: python

   Now in Progress - Data Preprocessing Ensemble Iteration: Estimate about 0
  Total combinations: 64
       diagnosis    Size_3  area_mean  compactness_mean  concave points_mea
  0            1  1.290564   1.151477          1.730765             1.63873
  1            1 -1.423416   1.823311         -0.679975             0.53514
  2            1 -0.066426   1.823311          1.163585             1.63873
  3            1  1.290564  -1.011406          1.730765             1.63873
  4            1 -0.066426   1.823311          0.548763             1.61037
  ..         ...       ...        ...               ...                  ..
  281          0 -0.066426  -0.866487         -1.300016            -0.80503
  282          1 -0.066426   1.657991          0.807396             1.30604
  283          1 -0.066426   0.462408          1.624135             1.17625
  284          0 -0.066426  -0.552378         -0.290663            -0.60750
  285          0  1.290564  -0.649460         -1.300016            -1.25679
  [286 rows x 26 columns]

..


Features selection for a regression problem using autoFS:
---------------------------------------------------------

  Demo Code:

.. code-block:: python

   import pandas as pd
   from dynapipe.autoFS import dynaFS_reg

   tr_features = pd.read_csv('./data/regression/train_features.csv')
   tr_labels = pd.read_csv('./data/regression/train_labels.csv')
   
   # Set input_form_file = False, when label values are array. Select 'True' from Pandas dataframe.
   reg_fs_demo = dynaFS_reg( fs_num = 5,random_state = 13,cv = 5,input_from_file = True)
   # Select detail_info = True, when you want to see the detail of the iteration
   reg_fs_demo.fit(tr_features,tr_labels,detail_info = False)
..

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
..

Model selection for a classification problem using autoCV:
----------------------------------------------------------

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
   clf_cv_demo.fit(tr_features,tr_labels,detail_info = False)
   
   models = {}
   for mdl in ['lgr','svm','mlp','rf','ada','gb','xgb']:
       models[mdl] = joblib.load('./pkl/{}_clf_model.pkl'.format(mdl))

   for name, mdl in models.items():
       evaluate_clf_model(name, mdl, val_features, val_labels)
..

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
..

Custom estimators & parameters setting for for autoCV:
------------------------------------------------------

  Currently, there're 3 methods in *utilis_fun* module - *reset_parameters*, *update_parameters*, and *export_parameters*.

  - *update_parameters* method is used to modify the default parameter settings for models selection module (autoCV).

     i.e. When you want to modify the support vector machine classifier, with new penalty "C" and "kernel" values, the code line below will achieve that.

.. code-block:: python

 update_parameters(mode = "cls", estimator_name = "svm", C=[0.1,0.2],kernel=["linear"])
..

  - *export_parameters* method can help you export the currnt default parameter settings as 2 csv files named "exported_cls_parameters.csv" and "exported_reg_parameters.csv". You can find them in the *./exported* folder of you current work dictionary.

.. code-block:: python

 export_parameters()
..

  - *reset_parameters* method can reset the default parameter settings to the package's original default settings. Just add this code line will work:

.. code-block:: python

 reset_parameters()
..

Dynamic Pipeline building using autoPipe:
-----------------------------------------

  Demo Code:

.. code-block:: python

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
..

 Output:

.. code-block:: python

  	Dataset	   Model_Name    Best_Parameters	 Accuracy	Precision	Recall	Latency
  1	Dataset_0	svm	[('C', 0.1), ('kernel', 'linear')]	  0.930 0.889 0.96 3.0
  6	Dataset_0	xgb	[('learning_rate', 1), ('max_depth', 2), ('n_estimators', 50), ('random_state', 13)]	0.912	0.955	0.84	2.0
  40	Dataset_5	gb	[('learning_rate', 1), ('max_depth', 2), ('n_estimators', 50), ('random_state', 13)]	0.895	0.913	0.84	2.0
  31	Dataset_4	rf	[('max_depth', 2), ('n_estimators', 50), ('random_state', 13)]	0.877	0.821	0.92	12.0
  51	Dataset_7	mlp	[('activation', 'relu'), ('hidden_layer_sizes', (10,)), ('learning_rate', 'constant'), ('random_state', 13), ('solver', 'sgd')]	0.772	0.875	0.56	4.0

..
