#!/usr/bin/env python

import time, sys
from IPython.display import clear_output
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from time import time
import numpy as np
from pandas.io.json import json_normalize
import os
import json

def delete_old_log_files(directory = None, delete_flag = False, logger = None, extension_list = None,filename_list = None, log_ts = None):
    file_list = os.listdir(directory)
    Test_comment = '-' * 20 * 3
    logger.info("Copyright All Reserved by Tony Dong | e-mail: tonyleidong@gmail.com ")
    logger.info("Official Documentation: https://dynamic-pipeline.readthedocs.io")
    logger.info(Test_comment)
    if delete_flag:
        logger.info("All previous logfiles will be deleted, when DELETE_FLAG is set to True.")

        for item in file_list:
            ext_flag = [item.startswith(i) for i in filename_list]
            if np.sum(ext_flag) and (log_ts not in item):
                os.remove(os.path.join(directory, item))
                logger.info(f"Deleted file:{item}")

    return None

def clear():
    os.system( 'cls' )

def update_progress(progress, clear_flag = False,process_name = None,time_est = None):
    if (clear_flag):
        clear()
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    if(time_est is None):
        text = "Now in Progress - " +process_name+ ": [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block),progress * 100)
    else:
        text = "Now in Progress - " +process_name+ ": Estimate about "+str(time_est)+" minutes left " +" [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block),progress * 100)
    
    print(text)

def pipeline_splitting_rule(val_size = 0.2, test_size = 0.2, random_state = 13):
    custom_val_size,custom_size,custom_random_state = val_size, test_size, random_state
    return(custom_val_size,custom_size,custom_random_state)

def data_splitting_tool(feature_cols = None, label_col = None ,val_size = 0.2, test_size = 0.2, random_state = 13):
    if (val_size != ''):
        total_test_size = val_size + test_size
        test_ratio = test_size/total_test_size
        X_train, X_test, y_train, y_test = train_test_split(feature_cols, label_col, test_size = total_test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio, random_state=random_state)
        return(X_train, y_train, X_val,y_val, X_test, y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(feature_cols, label_col, test_size = test_size, random_state=random_state)
        return(X_train, y_train, X_test, y_test)

def reset_parameters():
    try:
        json_p = os.path.join(os.path.dirname(__file__), 'reset_parameters.json')
        with open(json_p,'r') as d_file:
            para = json.load(d_file)
        json_p = os.path.join(os.path.dirname(__file__), 'parameters.json')
        w_file = open(json_p, "w",encoding='utf-8')
        w_file. truncate(0)
        json.dump(para, w_file)
        w_file.close()
        print('Done with the parameters reset.')
    except:
        print('Failed to reset the parameters.')

def update_parameters(mode = str(None), estimator_name = str(None), **kwargs):
    try:
        json_p = os.path.join(os.path.dirname(__file__), 'parameters.json')
        with open(json_p,'r',encoding='utf-8') as d_file:
            para = json.load(d_file)
        print(f"Previous Parameters are: {para[mode][estimator_name]}")
        para[mode][estimator_name] = kwargs
        print(f"Current Parameters are updated as: {para[mode][estimator_name]}")
        json_p = os.path.join(os.path.dirname(__file__), 'parameters.json')
        w_file = open(json_p, "w",encoding='utf-8')
        json.dump(para, w_file)
        w_file.close()
        print('Done with the parameters update.')
    except:
        print('Failed to update the parameters.')

def export_parameters():
    exp_folder = os.path.join(os.getcwd(),'exported')
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    try:
        json_p = os.path.join(os.path.dirname(__file__), 'parameters.json')
        with open(json_p,"r") as d_file:
            para = json.load(d_file)
        para_pd = pd.json_normalize(para["cls"])
        para_pd.to_csv(os.path.join(exp_folder,"exported_cls_parameters.csv"),index = False)
        para_pd = pd.json_normalize(para["reg"])
        para_pd.to_csv(os.path.join(exp_folder,"exported_reg_parameters.csv"),index = False)
        print('Done with the parameters setting file export.')
    except:
        print('Failed to export the parameters file.')

