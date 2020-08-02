"""Modules"""

#!/usr/bin/env python

import itertools
from dynapipe.funcPP import PPtools
import pandas as pd
import joblib
import datetime
import numpy as np
from time import time
from dynapipe.utilis_func import update_progress,delete_old_log_files
import warnings
import os

path = os.getcwd()

def warn(*args, **kwargs):
    pass
warnings.warn = warn

import logging

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
Test_case = f'Dynamic Pipeline - autoCV - Auto PreProcessing :: {LOG_TS}'
Test_comment = '-' * len(Test_case) * 3
Start_log = '#' * len(Test_case) * 3
logger.info(Start_log)
logger.info(Test_case)
logger.info(Start_log)
delete_old_log_files(directory = logs_folder ,delete_flag = DELETE_FLAG, logger = logger, extension_list = ['.log'],filename_list = ['autoPP_log'],log_ts = LOG_TS)
logger.info(Test_comment)

class dynaPreprocessing:
    """Automated feature preprocessing including imputation, winsorization, encoding, and scaling in ensemble algorithms, to generate permutation input datasets for further pipeline components.
    Parameters
    ----------
    custom_parameters: dictionary
        Custom parameters settings input - Default: None.
        NOTE - default_parameters = {
            "scaler" : ["None", "standard", "minmax", "maxabs", "robust"],
            "encode_band" : [10],
            "low_encode" : ["onehot","label"], 
            "high_encode" : ["frequency", "mean"],
            "winsorizer" : [(0.01,0.01),(0.05,0.05)],
            "sparsity" : [0.40],
            "cols" : [30]
            }
    label_col: str
        Name of label column - Default: None.
    model_type: str
        "reg" for regression problem or "cls" for classification problem - Default: "reg".
    export_output_files: bool
        Export qualified permutated datasets to ./df_folder - Default: False.
    
    Example
    -------
    See Demos -> Demo 1
    
    References
    ----------
    None
    """
    def __init__(self, custom_parameters = None, label_col = None, model_type = "reg",export_output_files = False):

        default_parameters = {
            "scaler" : ["None", "standard", "minmax", "maxabs", "robust"],
            "encode_band" : [10],
            "low_encode" : ["onehot","label"], 
            "high_encode" : ["frequency", "mean"],
            "winsorizer" : [(0.01,0.01),(0.05,0.05)],
            "sparsity" : [0.40],
            "cols" : [30]
            }
        if(custom_parameters is None):
            self.parameters = default_parameters
        else:
            self.parameters = custom_parameters
        
        self.model_type = model_type
        self.export_output_files = export_output_files
        self.label_col = label_col

    def fit(self, input_data = None):
        """Fits and transforms a pandas dataframe to non-missing values, outlier excluded, categories encoded and scaled datasets by all algorithms permutation.
        Parameters
        ----------
        input_data : pandas dataframe, shape = [n_samples, n_features]
            NOTE - The input_data should be the datasets after basic data cleaning & well feature deduction, the more features involve will result in more columns permutation outputs. 
        Returns
        -------
        DICT_PREP_DF : dictionary
            Each key is the # of the output preprocessed dataset, each value stores the dataset
        DICT_PREP_INFO : dictionary
            Dictionary for reference. Each key is the # of the output preprocessed dataset, each value stores the column names of the dataset
        """
                        
        if (self.export_output_files):
            df_folder = os.path.join(os.getcwd(),'dfs')
            if not os.path.exists(df_folder):
                os.makedirs(df_folder)
            for l in os.listdir(df_folder):
                os.remove(os.path.join(df_folder,l))
        DICT_DFS={}
        for i in range(len(self.parameters.get("winsorizer"))):
            pp = PPtools(label_col = self.label_col, data = input_data, model_type = self.model_type)
            pp.split_category_cols()
            initial_num_cols = pp.num_df.columns
            pp.impute_tool()
            pp.winzorize_tool(lower_ban = self.parameters.get("winsorizer")[i][0],upper_ban = self.parameters.get("winsorizer")[i][1])
            winsorized_df_cols_list = list(pp.num_df.columns)
            encoded_cols_list = {}
            for col in pp.cat_df.columns:
                encoded_cols_list[col] = []
                if(pp.cat_df[col].nunique() < self.parameters.get("encode_band")[0]):
                    for en_type in self.parameters.get("low_encode"):
                        encoded_col = pp.encode_tool(en_type = en_type ,category_col = col)
                        encoded_cols_list[col].append(list(encoded_col.columns))
                        pp.num_df = pd.concat([pp.num_df,encoded_col],axis = 1)                       

                if(pp.cat_df[col].nunique() >= self.parameters.get("encode_band")[0]):
                    for en_type in self.parameters.get("high_encode"):
                        encoded_col = pp.encode_tool(en_type = en_type ,category_col = col)
                        encoded_cols_list[col].append(list(encoded_col.columns))                       
                        pp.num_df = pd.concat([pp.num_df,encoded_col],axis = 1)

            args_list = []
            for key in encoded_cols_list.keys():
                args_list.append(encoded_cols_list[key])
            iters_combined = itertools.product(*args_list)
            loop_num = 1
            total_loop = len(list(iters_combined))                            
            for number, combination in enumerate(itertools.product(*args_list)):
                start_time = time()
                combined_cols_list = []
                combined_cols_list.append(winsorized_df_cols_list)
                
                for ele in list(combination):
                    combined_cols_list.append(ele)
                
                combined_cols_list = [item for sublist in combined_cols_list for item in sublist]
                encoded_df = pp.num_df[pp.num_df.columns.intersection(combined_cols_list)]
                encoded_df = pp.remove_zero_col_tool(encoded_df)
                category_sparsity_score = pp.sparsity_tool(encoded_df[encoded_df.columns.difference(list(initial_num_cols))])
                if (category_sparsity_score > self.parameters["sparsity"][0]) and ((len(encoded_df.columns)+1)<=self.parameters["cols"][0]):
                    logger.info(Test_comment)
                    logger.info(f"Current Running Dataset No. {number} :")
                    if (self.export_output_files):
                        temp_dfs = os.path.join(df_folder, f"winsor_{i}_{number}.csv")
                        encoded_df.to_csv(temp_dfs, index = False)
                    

                    for sca in self.parameters["scaler"]:
                        DICT_DFS[f"winsor_{i}-Scaler_{sca}-Dataset_{number}"] = pd.concat([pp.data[self.label_col], pp.scale_tool(df = encoded_df,sc_type = sca)],axis = 1)
                        logger.info(f">>> winsorized_Strategy is {i}")
                        logger.info(f">>> Scaler stragety is {sca}")
                        logger.info(f">>> Encoding strategy: {list(combination)}")
                        logger.info(f">>> Total columns with label column is: {len(list(encoded_df.columns))+1}")
                        logger.info(f">>> Encoded Category Columns' Sparsity Score: {str(category_sparsity_score)}")
                time_est = round(((time()-start_time)/60)*(total_loop - loop_num),4)
                update_progress(loop_num/total_loop, clear_flag = True, process_name = "Data Preprocessing Ensemble Iteration", time_est = time_est)
                loop_num += 1
        DICT_PREP_INFO = {}
        DICT_PREP_DF = {}
        for number, key in enumerate(DICT_DFS.keys()):
            DICT_PREP_INFO["Dataset_"+str(number)] = key.split("Dataset_",1)[0] + "- Encoded Features:" + str(list(DICT_DFS[key].columns))
        for number, key in enumerate(DICT_DFS.keys()):
            DICT_PREP_DF["Dataset_"+str(number)] = DICT_DFS[key]
        
        return(DICT_PREP_DF,DICT_PREP_INFO)
      


