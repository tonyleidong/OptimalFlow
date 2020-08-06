#!/usr/bin/env python

from dynapipe.estimatorCV import clf_cv,reg_cv
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dynapipe.utilis_func import update_progress,delete_old_log_files
import joblib
import datetime
import numpy as np
from time import time
import warnings
import os
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
Test_case = f'Dynamic Pipeline - autoCV - Auto Model Selection w/ Cross Validation :: {LOG_TS}'
Test_comment = '-' * len(Test_case) * 3
Start_log = '#' * len(Test_case) * 3
logger.info(Start_log)
logger.info(Test_case)
logger.info(Start_log)
delete_old_log_files(directory = logs_folder ,delete_flag = DELETE_FLAG, logger = logger, extension_list = ['.log'],filename_list = ['autoCV_log'],log_ts = LOG_TS)
logger.info(Test_comment)


def warn(*args, **kwargs):
    pass

def print_results(results,in_pipeline = False):
    if (not in_pipeline):
        print('Best Parameters: {}\n'.format(results.best_params_))
        print('Best CV Score: {}\n'.format(results.best_score_))
    logger.info('Best Paramaters: {}\n'.format(results.best_params_))
    logger.info('Best CV Score: {}\n'.format(results.best_score_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        logger.info('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

class evaluate_model:
    """This class implements model evaluation and return key score results.
    
    Parameters
    ----------
    model_type : str, default = None
        Value in ["reg","cls"]. The "reg" for regression problem, and "cls" for classification problem.

    in_pipeline : bool, default = False
        Should be set to "True" when using autoPipe module to build Pipeline Cluster Traveral Experiments.

    Example
    -------
    
    .. [Example]: 
    
    References
    ----------
    
    """
    def __init__(self,model_type = None, in_pipeline = False):
        self.model_type = model_type
        self.in_pipeline = in_pipeline
        optimal_scores = []

    def fit(self,name = None, model = None, features = None, labels = None):
        """Model evaluation with all models by the validate datasets.
        
        Parameters
        ----------
        name : str, default = None
            Estimator name.
        model : pkl, default = None
            Model needs to evaluate. Needs pkl file as input when in_pipeline = "False"; otherwise, should use DICT_EST[estimator name] as the input here.
        features : df, default = None
            Validate features columns. ( NOTE: In the Pipeline Cluster Traversal Experiments, the features columns should be from the same pipeline dataset).
        labels : df ,default = None
            Validate label column. 
            ( NOTE: In the Pipeline Cluster Traversal Experiments, the label column should be from the same pipeline dataset).        
        Returns
        -------
        optimal_scores : list
            When model_type = "cls", will return [name,accuracy,precision,recall,latency] info of model validation results.
            when model_type = "reg", will return [name,R-squared,MAE,MSE,RMSE,latency] info of model validation results.
            
        """
        if (self.model_type == "cls"):
            start = time()
            pred = model.predict(features)
            end = time()
            accuracy = round(accuracy_score(labels, pred), 3)
            precision = round(precision_score(labels, pred), 3)
            recall = round(recall_score(labels, pred), 3)
            latency = round((end - start)*1000, 1)
            optimal_scores = [name,accuracy,precision,recall,latency]
            if(not self.in_pipeline):
                print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}s'.format(name,accuracy,precision,recall,latency))
                logger.info('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}s'.format(name,accuracy,precision,recall,latency))
            if(self.in_pipeline):
                logger.info('>>> {} Modle Validation Results -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}s'.format(name,accuracy,precision,recall,latency))
        if (self.model_type == "reg"):
            start = time()
            pred = model.predict(features)
            end = time()
            R2 = round(metrics.r2_score(labels, pred),3)
            MAE = round(metrics.mean_absolute_error(labels, pred),3)
            MSE = round(metrics.mean_squared_error(labels, pred),3)
            RMSE = round(metrics.mean_squared_error(labels, pred),3)
            latency = round((end - start)*1000, 1)
            optimal_scores = [name,R2,MAE,MSE,RMSE,latency]
            if(not self.in_pipeline):
                print(f'{name} -- R^2 Score: {R2} / Mean Absolute Error: {MAE} / Mean Squared Error: {MSE} / Root Mean Squared Error: {RMSE} / Latency: {latency}s')
                logger.info(f'{name} -- R^2 Score: {R2} / Mean Absolute Error: {MAE} / Mean Squared Error: {MSE} / Root Mean Squared Error: {RMSE} / Latency: {latency}s')
            if(self.in_pipeline):
                logger.info(f'>>> {name} Model Validation Results -- R^2 Score: {R2} / Mean Absolute Error: {MAE} / Mean Squared Error: {MSE} / Root Mean Squared Error: {RMSE} / Latency: {latency}s')
        return(optimal_scores)

class dynaClassifier:
    """This class implements classification model selection with hyperparameters grid search and cross-validation.
    
    Parameters
    ----------
    random_state : int, default = None
        Random state value.
    
    cv : int, default = None
        # of folds for cross-validation.

    in_pipeline : bool, default = False
        Should be set to "True" when using autoPipe module to build Pipeline Cluster Traveral Experiments.
    
    input_from_file : bool, default = True
        When input dataset is df, needs to set "True"; Otherwise, i.e. array, needs to set "False".

    Example
    -------
    .. [Example] https://dynamic-pipeline.readthedocs.io/en/latest/demos.html#model-selection-for-a-classification-problem-using-autocv
    
    References
    ----------
    None
    """
    def __init__(self,random_state = 13,cv_num = 5,in_pipeline = False, input_from_file = True):
        self.random_state =random_state
        self.cv_num = cv_num
        self.input_from_file = input_from_file
        self.in_pipeline = in_pipeline
        self.DICT_EST = {}
    def fit(self,tr_features = None,tr_labels = None):
        """Fit and train datasets with classification hyperparameters GridSearch and CV across multiple estimators. Module will Auto save trained model as {estimator_name}_clf_model.pkl file to ./pkl folder.
        Parameters
        ----------

        features : df, default = None
            Train features columns. ( NOTE: In the Pipeline Cluster Traversal Experiments, the features columns should be from the same pipeline dataset).
        labels : df ,default = None
            Train label column. 
            ( NOTE: In the Pipeline Cluster Traversal Experiments, the label column should be from the same pipeline dataset).        
        Returns
        -------
            cv_num : int
                # of fold for cross-validation.
            DICT_EST : dictionary
                key is the name of estimators, value is the ralated trained model
            
            NOTE - Trained model auto save function only avalable when in_pipeline = "False".
            NOTE - Log records will generate and save to ./logs folder automatedly.
        """
        warnings.warn = warn
        if(self.input_from_file):
            tr_labels = tr_labels.values.ravel()
        clf = clf_cv(cv_val = self.cv_num,random_state = self.random_state)
        estimators = ['lgr','svm','mlp','rf','ada','gb','xgb']
        loop_num = 1
        total_loop = len(estimators)
        if(not self.in_pipeline):
            pkl_folder = os.path.join(os.getcwd(),'pkl')
            if not os.path.exists(pkl_folder):
                os.makedirs(pkl_folder)

        for est in estimators:
            start_time = time()
            logger.info(Test_comment)
            logger.info(f"Current Running:" + est +" estimator")
            try:
                cv_est = getattr(clf, est)()
                cv_est.fit(tr_features,tr_labels)
                if(not self.in_pipeline):
                    model_name = os.path.join(pkl_folder, f'{est}_clf_model.pkl')
                    joblib.dump(cv_est.best_estimator_, model_name)
                    time_est = round(((time()-start_time)/60)*(total_loop - loop_num),4)
                    update_progress(loop_num/total_loop, clear_flag = True, process_name = "Model Selection w/ Cross-validation",time_est= time_est)
                print(f"\n    *DynaPipe* autoCV Module ===> {est}_CrossValidation with {self.cv_num} folds:")
                print_results(cv_est,self.in_pipeline)
                self.DICT_EST[est] = cv_est

                logger.info(f"This estimator executed {round((time()-start_time)/60,4)} minutes")
                loop_num += 1
            except:
                print(est+" estimator is not availible.")
                if(not self.in_pipeline):
                    time_est = round(((time()-start_time)/60)*(total_loop - loop_num),4)
                    update_progress(loop_num/total_loop, clear_flag = True, process_name = "Model Selection w/ Cross-validation",time_est= time_est)
                logger.info(f"This estimator executed {round((time()-start_time)/60,4)} minutes")
                loop_num += 1
                pass
        return(self.cv_num,self.DICT_EST)
class dynaRegressor:
    """This class implements regression model selection with with hyperparameters grid search and cross-validation. Module will Auto save trained model as {estimator_name}_reg_model.pkl file to ./pkl folder.
    
    Parameters
    ----------
    random_state : int, default = None
        Random state value.
    
    cv : int, default = None
        # of folds for cross-validation.

    in_pipeline : bool, default = False
        Should be set to "True" when using autoPipe module to build Pipeline Cluster Traveral Experiments.
    
    input_from_file : bool, default = True
        When input dataset is df, needs to set "True"; Otherwise, i.e. array, needs to set "False".

    Example
    -------
    .. [Example]
    
    References
    ----------
    None
    """
    def __init__(self ,random_state = 25 ,cv_num = 5,in_pipeline = False, input_from_file = True):
        self.random_state =random_state
        self.cv_num = cv_num
        self.input_from_file = input_from_file
        self.in_pipeline = in_pipeline
        self.DICT_EST = {}

    def fit(self,tr_features = None,tr_labels = None):
        """Fit and train datasets with regression hyperparameters GridSearch and CV across multiple estimators.
        
        Parameters
        ----------

        features : df, default = None
            Train features columns. ( NOTE: In the Pipeline Cluster Traversal Experiments, the features columns should be from the same pipeline dataset).
        labels : df ,default = None
            Train label column. 
            ( NOTE: In the Pipeline Cluster Traversal Experiments, the label column should be from the same pipeline dataset).        
        Returns
        -------
            cv_num : int
                # of fold for cross-validation.
            DICT_EST : dictionary
                key is the name of estimators, value is the ralated trained model.

            NOTE - Trained model auto save function only avalable when in_pipeline = "False".
            NOTE - Log records will generate and save to ./logs folder automatedly.
        """
        if(self.input_from_file):
            tr_labels = tr_labels.values.ravel()
        reg = reg_cv(cv_val = self.cv_num,random_state = self.random_state)
        estimators = ['lr','knn','tree','svm','mlp','rf','gb','ada','xgb']
        if (not self.in_pipeline):
            pkl_folder = os.path.join(os.getcwd(),'pkl')
            if not os.path.exists(pkl_folder):
                os.makedirs(pkl_folder)
        loop_num = 1
        total_loop = len(estimators)
        
        for est in estimators:
            start_time = time()
            logger.info(Test_comment)
            logger.info(f"Current Running:" + est +" estimator")
            try:
                cv_est = getattr(reg, est)()
                cv_est.fit(tr_features,tr_labels)
                if (not self.in_pipeline):
                    model_name = os.path.join(pkl_folder, f'{est}_reg_model.pkl')
                    joblib.dump(cv_est.best_estimator_, model_name)
                    time_est = round(((time()-start_time)/60)*(total_loop - loop_num),4)
                    update_progress(loop_num/total_loop, clear_flag = False, process_name = "Model Selection w/ Cross-validation",time_est= time_est)
                
                print(f"\n    *DynaPipe* autoCV Module ===> {est} model CrossValidation with {self.cv_num} folds:")
                print_results(cv_est,self.in_pipeline)
                self.DICT_EST[est] = cv_est

                logger.info(f"This estimator executed {round((time()-start_time)/60,4)} minutes")
                loop_num += 1
            except:
                print(est+" estimator is not availible.")
                if (not self.in_pipeline):
                    time_est = round(((time()-start_time)/60)*(total_loop - loop_num),4)
                    update_progress(loop_num/total_loop, clear_flag = True, process_name = "Model Selection w/ Cross-validation",time_est= time_est)
                logger.info(f"This estimator executed {round((time()-start_time)/60,4)} minutes")
                loop_num += 1
                pass
        return(self.cv_num,self.DICT_EST)
