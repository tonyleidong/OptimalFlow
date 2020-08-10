#!/usr/bin/env python 

from optimalflow.selectorFS import clf_fs,reg_fs
from sklearn.metrics import accuracy_score, precision_score, recall_score
from optimalflow.utilis_func import update_progress,delete_old_log_files
import joblib
import datetime
import numpy as np
from time import time
from collections import Counter
import os
import warnings
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
Test_case = f'Optimal Flow - autoFS - Auto Feature Selection Module :: {LOG_TS}'
Test_comment = '-' * len(Test_case) * 3
Start_log = '#' * len(Test_case) * 3
logger.info(Start_log)
logger.info(Test_case)
logger.info(Start_log)
delete_old_log_files(directory = logs_folder ,delete_flag = DELETE_FLAG, logger = logger, extension_list = ['.log'],filename_list = ['autoFS_log'],log_ts = LOG_TS)
logger.info(Test_comment)

def warn(*args, **kwargs):
    pass

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def rank_fs_result(clf_sel_result = None,tr_features = None):
    mask = clf_sel_result.get_support()
    fs_features = []
    feature_names = list(tr_features.columns.values)
    for bool, feature in zip(mask, feature_names):
        if bool:
            fs_features.append(feature)
    return(fs_features)

class dynaFS_clf:
    """This class implements feature selection for classification problem.
    
    Parameters
    ----------
    fs_num : int, default = None
        Set the # of features want to select out.
    
    random_state : int, default = None
        Random state value.
    
    cv : int, default = None
        # of folds for cross-validation.

    in_pipeline : bool, default = False
        Should be set to "True" when using autoPipe module to build Pipeline Cluster Traveral Experiments.
    
    input_from_file : bool, default = True
        When input dataset is dataframe, needs to set "True"; Otherwise, i.e. array, needs to set "False".

    Example
    -------
    .. []
    
    References
    ----------
    None
    """
    def __init__(self, fs_num = None ,random_state = None,cv = None, in_pipeline = False, input_from_file = True):
        self.fs_num = fs_num
        self.random_state = random_state
        self.cv = cv
        self.input_from_file = input_from_file
        self.in_pipeline = in_pipeline

    def fit(self,tr_features,tr_labels):
        """Fits and transforms a dataframe with built-in algorithms, to select top features.
        
        Parameters
        ----------
        tr_features : df, default = None
            Train features columns. 
            (NOTE: In the Pipeline Cluster Traversal Experiments, the features columns should be from the same pipeline dataset).
        tr_labels : array/df, default = None
            Train label column, when input_from_file = True, must be pandas datframe.
            (NOTE: In the Pipeline Cluster Traversal Experiments, the label column should be from the same pipeline dataset).
        
        Returns
        -------
        fs_num : int
            # of top features has been select out.
        fs_results : array
            Selected & ranked top feature names.

        NOTE - Log records will generate and save to ./logs folder automatedly.
        """
        warnings.warn = warn
        if(self.input_from_file):
            tr_labels = tr_labels.values.ravel()
        
        clf = clf_fs(fs_num = self.fs_num ,random_state = self.random_state,cv = self.cv)
        selectors = ['kbest_f','kbest_chi2','rfe_lr','rfe_svm','rfe_tree','rfe_rf','rfecv_svm','rfecv_tree','rfecv_rf']
                    
        loop_num = 1
        total_loop = len(selectors)
        selected_features = [] 
        for selector in selectors:
            start_time = time()
            if (not self.in_pipeline):
                logger.info(Test_comment)
                logger.info(f"Current Running:" + selector +" selector")
            try:
                clf_selector = getattr(clf, selector)()
                clf_sel_result = clf_selector.fit(tr_features,tr_labels)
                fs_feature = rank_fs_result(clf_sel_result,tr_features.head(1))

                selected_features.extend(fs_feature)
                if (not self.in_pipeline):
                    print(f'\n      *optimalflow* autoFS Module ===> Selector {selector} gets outputs: {fs_feature}')
                    update_progress(loop_num/total_loop,process_name = "Feature Selection Iteration")
                    logger.info(f"This selector executed {round((time()-start_time)/60,4)} minutes")
                loop_num += 1

            except:
                if (not self.in_pipeline):
                    print(selector+" selector is not availible.")
                    update_progress(loop_num/total_loop)
                    logger.info(f"This selector executed {round((time()-start_time)/60,4)} minutes")
                loop_num += 1
                pass
        
        counts = Counter(selected_features)
        fs_results = sorted(selected_features, key=lambda x: -counts[x])
        fs_results = unique(fs_results)[:self.fs_num]
        if (not self.in_pipeline):
            print(f"The optimalflow autoFS identify the top {self.fs_num} important features for classification are: {fs_results}.")
            logger.info(f"The optimalflow autoFS identify the top {self.fs_num} important features for classification are: {fs_results}.")
        return(self.fs_num,fs_results)           

class dynaFS_reg:
    """This class implements feature selection for regression problem.
    
    Parameters
    ----------
    fs_num : int, default = None
        Set the # of features want to select out.
    
    random_state : int, default = None
        Random state value.
    
    cv : int, default = None
        # of folds for cross-validation.

    in_pipeline : bool, default = False
        Should be set to "True" when using autoPipe module to build Pipeline Cluster Traveral Experiments.
    
    input_from_file : bool, default = True
        When input dataset is dataframe, needs to set "True"; Otherwise, i.e. array, needs to set "False".

    Example
    -------
    
    .. [Example] https://Optimal-Flow.readthedocs.io/en/latest/demos.html#features-selection-for-a-regression-problem-using-autoFS

    References
    ----------
    None
    """

    def __init__(self, fs_num = None ,random_state = None,cv = None,in_pipeline = False, input_from_file = True):
        self.fs_num = fs_num
        self.random_state = random_state
        self.cv = cv
        self.input_from_file = input_from_file
        self.in_pipeline = in_pipeline

    def fit(self,tr_features,tr_labels):
        """Fits and transforms a dataframe with built-in algorithms, to select top features.
        
        Parameters
        ----------
        tr_features : df, default = None
            Train features columns. 
            (NOTE: In the Pipeline Cluster Traversal Experiments, the features columns should be from the same pipeline dataset).
        tr_labels : array/df, default = None
            Train label column, when input_from_file = True, must be pandas datframe.
            (NOTE: In the Pipeline Cluster Traversal Experiments, the label column should be from the same pipeline dataset).
             
        Returns
        -------
        fs_num : int
            # of top features has been select out.
        fs_results : array
            Selected & ranked top feature names.
        
        NOTE - Log records will generate and save to ./logs folder automatedly.
        """
        if(self.input_from_file):
            tr_labels = tr_labels.values.ravel()

        reg = reg_fs(fs_num = self.fs_num ,random_state = self.random_state,cv = self.cv)
        selectors = ['kbest_f','rfe_svm','rfe_tree','rfe_rf','rfecv_svm','rfecv_tree','rfecv_rf']
                    
        loop_num = 1
        total_loop = len(selectors)
        selected_features = [] 
        for selector in selectors:
            start_time = time()
            if (not self.in_pipeline):
                logger.info(Test_comment)
                logger.info(f"Current Running:" + selector +" selector")
            try:
                reg_selector = getattr(reg, selector)()
                reg_sel_result = reg_selector.fit(tr_features,tr_labels)
                fs_feature = rank_fs_result(reg_sel_result,tr_features.head(1))
                
                selected_features.extend(fs_feature)
                if (not self.in_pipeline):
                    update_progress(loop_num/total_loop,process_name = "Feature Selection Iteration")
                    print(f'\n      *optimalflow* autoFS Module ===> Selector {selector} gets outputs: {fs_feature}')
                    logger.info(f"This selector executed {round((time()-start_time)/60,4)} minutes")
                loop_num += 1

            except:
                if (not self.in_pipeline):
                    print(selector+" selector is not availible.")
                    update_progress(loop_num/total_loop)
                    logger.info(f"This selector executed {round((time()-start_time)/60,4)} minutes")
                loop_num += 1
                pass
        
        counts = Counter(selected_features)
        fs_results = sorted(selected_features, key=lambda x: -counts[x])
        fs_results = unique(fs_results)[:self.fs_num]
        if (not self.in_pipeline):
            print(f"The optimalflow autoFS identify the top {self.fs_num} important features for regression are: {fs_results}.")
            logger.info(f"The optimalflow autoFS identify the top {self.fs_num} important features for regression are: {fs_results}.")
        return(self.fs_num,fs_results)   
