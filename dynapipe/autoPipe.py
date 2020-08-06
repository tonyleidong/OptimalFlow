#!/usr/bin/env python 

import pandas as pd 
from dynapipe.funcPP import PPtools
from dynapipe.autoPP import dynaPreprocessing
from dynapipe.utilis_func import data_splitting_tool,delete_old_log_files,update_progress
from dynapipe.autoFS import dynaFS_clf,dynaFS_reg
from dynapipe.autoCV import evaluate_model,dynaClassifier,dynaRegressor
import datetime
import numpy as np
from time import time
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def warn(*args, **kwargs):
    pass
warnings.warn = warn
import logging

path = os.getcwd()

LOG_TS = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
logs_folder = os.path.join(os.getcwd(),'logs')
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)
log_name = os.path.join(logs_folder, f'{os.path.basename(__file__).split(".")[0]}_log_{LOG_TS}.log')

LOG_LEVEL = logging.DEBUG
DELETE_FLAG = True
TS = time()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s','%d/%m %H:%M:%S')
fh = logging.FileHandler(filename = log_name)
fh.setLevel(LOG_LEVEL)
fh.setFormatter(formatter)
logger.addHandler(fh)
Test_case = f'Dynamic Pipeline - autoCV - Auto Pipe Connector :: {LOG_TS}'
Test_comment = '-' * len(Test_case) * 3
Start_log = '#' * len(Test_case) * 3
logger.info(Start_log)
logger.info(Test_case)
logger.info(Start_log)
delete_old_log_files(directory = logs_folder ,delete_flag = DELETE_FLAG, logger = logger, extension_list = ['.log'],filename_list = ['autoPipe_log'],log_ts = LOG_TS)
logger.info(Test_comment)


class autoPipe:
    """This class is to build Pipeline Cluster Traversal Experiments.
    
    Parameters
    ----------
    steps: list, default = None
        List of (name, transform) tuples (implementing fit & transform) that are chained, in the order in which they are chained, with the last object a model evaluation function.
    
    Example
    -------
    
    .. [Example]: https://dynamic-pipeline.readthedocs.io/en/latest/demos.html#build-pipeline-cluster-traveral-experiments-using-autopipe
    
    
    References
    ----------
    
    None
    """
    def __init__(self,steps):
        self.step1 = steps[0][1]
        self.step2 = steps[1][1]
        self.step3 = steps[2][1]
        self.step4 = steps[3][1]
        self.step5 = steps[4][1]

    def fit(self,data):
        """Fits and transforms a chain of Dynamic Pipeline modules.
        
        Parameters
        ----------
        input_data : pandas dataframe, shape = [n_samples, n_features]
            
            NOTE: 
            The input_data should be the datasets after basic data cleaning & well feature deduction, the more features involve will result in more columns permutation outputs. 
        
        Returns
        -------
        DICT_PREP_INFO : dictionary
            Each key is the # of preprocessed dataset("Dataset_xxx" format, i.e. "Dataset_10"), each value stores an info string about what transforms applied.
            i.e. DICT_PREPROCESSING['Dataset_0'] stores value "winsor_0-Scaler_None-- Encoded Features:['diagnosis', 'Size_3', 'area_mean']", which means applied 1st mode of winsorization, none scaler applied, and the encoded columns names(shown the enconding approachs in the names)
        DICT_FEATURE_SELECTION_INFO : dictionary
            Each key is the # of preprocessed dataset, each value stores the name of features selected after the autoFS module.
        DICT_MODELS_EVALUATION : dictionary
            Each key is the # of preprocessed dataset, each value stores the model evaluation results with its validate dataset.
        DICT_DATA : dictionary
            Each key is the # of preprocessed dataset, and first level sub-key is the type of splitted sets(including 'DICT_Train','DICT_TEST',and'DICT_Validate').
            The second level sub-key is "X" for features and "y" for label, each value stores the datasets related to the keys(Pandas Dataframe format)
            i.e. DICT_DATA['Dataset_0']['DICT_TEST']["X"] is the train features of Dataset_0's test dataset
        models_summary : Pandas Dataframe
            Model selection results ranking table among all composits of preprocessed datasets, selected features and all posible models with optimal parameters. 
        
        NOTE - Log records will generate and save to ./logs folder automatedly.
        """
        dyna = self.step1
        DICT_PREP_DF,DICT_PREP_INFO = dyna.fit(input_data = data)
        print(f"Total combinations: {len(DICT_PREP_DF.keys())}")
        logger.info(f"Total combinations: {len(DICT_PREP_DF.keys())}")
        # Tracking the metrics values
        DICT_MODELS_EVALUATION = {}
        # Feature Selction tracking
        DICT_FEATURE_SELECTION_INFO = {}

        DICT_DATA = {}

        loop_num = 1
        total_loop = len(DICT_PREP_DF.keys())
        for number, key in enumerate(DICT_PREP_DF.keys()):
            combination_df = DICT_PREP_DF[key]
            start_time = time()
            logger.info(Test_comment)
            dataset_num = key.split("Dataset_",1)[1]
            logger.info(f"Current Running Preprocessed Dataset No. {dataset_num}:")
            features = combination_df.drop(dyna.label_col, axis=1)
            labels = combination_df[dyna.label_col]

            logger.info("[Features Before autoFS]: ")
            logger.info(list(features.columns))

            custom_val_size,custom_size,custom_random_state = self.step2
            X_train, y_train, X_val,y_val, X_test, y_test = data_splitting_tool(feature_cols = features, label_col = labels ,val_size = custom_val_size, test_size = custom_size, random_state = custom_random_state)
            tr_features = X_train
            tr_labels = y_train
            autoFS_module = self.step3
            fs_num, fs_results = autoFS_module.fit(tr_features,tr_labels)
            DICT_FEATURE_SELECTION_INFO["Dataset_"+ str(dataset_num)] = fs_results
            logger.info(f"[Results Report]:")
            logger.info(f">>> autoFS summary - This dataset has the top {fs_num} important features: {fs_results}.")

            tr_features = tr_features[list(fs_results)]
            tr_labels = tr_labels
            val_features = X_val[list(fs_results)]
            val_labels = y_val
            ts_features = X_test[list(fs_results)] 
            ts_labels = y_test

            DICT_PER_DATA = {
                "DICT_Train":{},
                "DICT_Validate":{},
                "DICT_TEST":{}
            }
            DICT_PER_DATA["DICT_Train"]["X"] = tr_features
            DICT_PER_DATA["DICT_Train"]["y"] = tr_labels
            DICT_PER_DATA["DICT_Validate"]["X"] = val_features
            DICT_PER_DATA["DICT_Validate"]["y"] = val_labels
            DICT_PER_DATA["DICT_TEST"]["X"] = ts_features
            DICT_PER_DATA["DICT_TEST"]["y"] = ts_labels

            DICT_DATA["Dataset_"+ str(dataset_num)] = DICT_PER_DATA

            autoCV_module = self.step4
            cv_num,DICT_EST = autoCV_module.fit(tr_features,tr_labels)
            for est in DICT_EST.keys():
                results = DICT_EST[est]
                logger.info(f">>> autoCV summary - {est} model CrossValidation with {cv_num} folds:")
                logger.info('     - Best Paramaters: {}\n'.format(results.best_params_))
                logger.info('     - Best CV Score: {}\n'.format(results.best_score_))
            
            evaluate_module = self.step5
            if (evaluate_module.model_type == "cls"):
                metrics_df = pd.DataFrame(columns=['Model_Name','Accuracy','Precision','Recall','Latency','Best_Parameters'])
            if (evaluate_module.model_type == "reg"):
                metrics_df = pd.DataFrame(columns=['Model_Name','R2','MAE','MSE','RMSE','Latency','Best_Parameters'])

            for est in DICT_EST.keys(): 
                optimal_scores = evaluate_module.fit(name = est, model = DICT_EST[est].best_estimator_,features = val_features, labels = val_labels)
                optimal_scores.append(str([i for i in DICT_EST[est].best_params_.items()]))

                if (evaluate_module.model_type == "cls"):
                    metrics_df = metrics_df.append(pd.DataFrame([optimal_scores],columns=['Model_Name','Accuracy','Precision','Recall','Latency','Best_Parameters']))
                    logger.info('>>> {} Modle Validation Results -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}s'.format(optimal_scores[0],optimal_scores[1],optimal_scores[2],optimal_scores[3],optimal_scores[4]))

                if (evaluate_module.model_type == "reg"):
                    metrics_df = metrics_df.append(pd.DataFrame([optimal_scores],columns=['Model_Name','R2','MAE','MSE','RMSE','Latency','Best_Parameters']))
                    logger.info('>>> {} Model Validation Results -- R^2 Score: {} / Mean Absolute Error: {} / Mean Squared Error: {} / Root Mean Squared Error: {} / Latency: {}s'.format(optimal_scores[0],optimal_scores[1],optimal_scores[2],optimal_scores[3],optimal_scores[4],optimal_scores[5]))

            
            DICT_MODELS_EVALUATION["Dataset_"+ str(dataset_num)] = metrics_df

            logger.info(f"Total executed {round((time()-start_time)/60,4)} minutes")
            time_est = round(((time()-start_time)/60)*(total_loop - loop_num),4)
            update_progress(loop_num/total_loop,clear_flag = True,process_name = "autoFS & autoCV Iteration",time_est = time_est)
            loop_num += 1

        dict_flow = DICT_MODELS_EVALUATION
        for key in dict_flow.keys():
            dict_flow[key]['Dataset'] = key
        if (evaluate_module.model_type == "cls"):
            models_summary = pd.concat([dict_flow[i] for i in dict_flow.keys()],ignore_index=True).sort_values(by=['Accuracy','Precision','Recall','Latency'], ascending=[False,False,False,True])
            models_summary = models_summary[["Dataset","Model_Name","Best_Parameters",'Accuracy','Precision','Recall','Latency']]
        if (evaluate_module.model_type == "reg"):
            models_summary = pd.concat([dict_flow[i] for i in dict_flow.keys()],ignore_index=True).sort_values(by=['Accuracy','Precision','Recall','Latency'], ascending=[False,False,False,True])
            models_summary = models_summary[["Dataset","Model_Name","Best_Parameters",'R2','MAE','MSE','RMSE','Latency']]          
        logger.info(Start_log)
        print(f"The top 5 Models with Best Performance Metrics:")
        print(models_summary.head(5))
        logger.info(f"The top 5 Models with Best Performance Metrics:")
        logger.info(models_summary.head(5))
        
        return(DICT_PREP_INFO,DICT_FEATURE_SELECTION_INFO,DICT_MODELS_EVALUATION,DICT_DATA,models_summary)

