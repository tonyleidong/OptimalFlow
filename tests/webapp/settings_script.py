import pandas as pd

from optimalflow.utilis_func import pipeline_splitting_rule, update_parameters,reset_parameters

import json
import os

json_path_s = os.path.join(os.path.dirname("./"), 'settings.json')
with open(json_path_s, encoding='utf-8') as data_file:
    para_data = json.load(data_file)
data_file.close()

custom_space = {
        "confirm_reset":para_data['confirm_reset']

}

reset_parameters()

update_parameters(mode = custom_space['mode'], estimator_name = custom_space['model'], custom_space['args'])


# update_parameters(mode = "cls", estimator_name = "svm", C=[0.1],kernel=["linear"])
# update_parameters(mode = "cls", estimator_name = "ada", n_estimators =[50],learning_rate=[1])
# update_parameters(mode = "cls", estimator_name = "rf", n_estimators =[50],max_depth=[2])
# update_parameters(mode = "cls", estimator_name = "gb", n_estimators =[50],max_depth=[2],learning_rate=[1])
# update_parameters(mode = "cls", estimator_name = "xgb", n_estimators =[50],max_depth=[2],learning_rate=[1])


if(custom_space['confirm_reset'] == "reset_default"):
    reset_parameters()

if(custom_space['confirm_reset'] == "reset_settings"):
    json_s = os.path.join(os.path.dirname("./"), 'reset_settings.json')
    with open(json_s,'r') as d_file:
        para = json.load(d_file)
    json_s = os.path.join(os.path.dirname("./"), 'settings.json')
    w_file = open(json_s, "w",encoding='utf-8')
    w_file. truncate(0)
    json.dump(para, w_file)
    w_file.close()