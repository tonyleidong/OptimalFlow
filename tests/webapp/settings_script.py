import pandas as pd

from optimalflow.utilis_func import pipeline_splitting_rule, update_parameters,reset_parameters

import json
import os

json_path_s = os.path.join(os.path.dirname("./"), 'settings.json')
with open(json_path_s, encoding='utf-8') as data_file:
    para_data = json.load(data_file)
data_file.close()

reset_flag = para_data['confirm_reset']

custom_space = {
        "cls_mlp":para_data['space_set']['cls']['mlp'],
        "cls_lr":para_data['space_set']['cls']['lgr'],
        "cls_svm":para_data['space_set']['cls']['svm'],
        "cls_ada":para_data['space_set']['cls']['ada'],
        "cls_xgb":para_data['space_set']['cls']['xgb'],
        "cls_rgcv":para_data['space_set']['cls']['rgcv'],
        "cls_rf":para_data['space_set']['cls']['rf'],
        "cls_gb":para_data['space_set']['cls']['gb'],
        "cls_lsvc":para_data['space_set']['cls']['lsvc'],
        "cls_hgboost":para_data['space_set']['cls']['hgboost'],
        "cls_sgd":para_data['space_set']['cls']['sgd'],
        "reg_lr":para_data['space_set']['reg']['lr'],
        "reg_svm":para_data['space_set']['reg']['svm'],
        "reg_mlp":para_data['space_set']['reg']['mlp'],
        "reg_ada":para_data['space_set']['reg']['ada'],
        "reg_rf":para_data['space_set']['reg']['rf'],
        "reg_gb":para_data['space_set']['reg']['gb'],
        "reg_xgb":para_data['space_set']['reg']['xgb'],
        "reg_tree":para_data['space_set']['reg']['tree'],
        "reg_hgboost":para_data['space_set']['reg']['hgboost'],
        "reg_rgcv":para_data['space_set']['reg']['rgcv'],
        "reg_cvlasso":para_data['space_set']['reg']['cvlasso'],
        "reg_huber":para_data['space_set']['reg']['huber'],
        "reg_sgd":para_data['space_set']['reg']['sgd']
}


try:
    if(reset_flag == "reset_default"):
        reset_parameters()
except:
    try:
        if(reset_flag == "reset_settings"):
            json_s = os.path.join(os.path.dirname("./"), 'reset_settings.json')
            with open(json_s,'r') as d_file:
                para = json.load(d_file)
            json_s = os.path.join(os.path.dirname("./"), 'settings.json')
            w_file = open(json_s, "w",encoding='utf-8')
            w_file. truncate(0)
            json.dump(para, w_file)
            w_file.close()
    except:
        try:            
            reset_parameters()
            for i in custom_space.keys():
                if custom_space[i]!={}:
                    model_type, algo_name=i.split('_')
                    update_parameters(mode = model_type,estimator_name=algo_name,**custom_space[i])
        except:
            print("Failed to Set Up the Searching Space, will Use the Default Settings!")