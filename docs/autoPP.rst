=============
autoPP Module
=============

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
    (NOTE: When you select 'None', might cause overfitting with too high R-Squared Score in Regression Problem)
        - "None" : None approach involve in scaling step 
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

dynapipePreprocessing
---------------------

.. autoclass:: optimalflow.autoPP.dynaPreprocessing
    :members:

PPtools
-------

.. autoclass:: optimalflow.funcPP.PPtools
    :members: