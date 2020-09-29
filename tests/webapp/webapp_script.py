import pandas as pd
from optimalflow.autoPipe import autoPipe
from optimalflow.funcPP import PPtools
from optimalflow.autoPP import dynaPreprocessing
from optimalflow.autoFS import dynaFS_clf,dynaFS_reg
from optimalflow.autoCV import evaluate_model,dynaClassifier,dynaRegressor,fastClassifier,fastRegressor
from optimalflow.utilis_func import pipeline_splitting_rule,reset_parameters,update_parameters

import json
import os

###------------------------ Estimators Searching Space Settings --------------------------------------
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
        "reg_sgd":para_data['space_set']['reg']['sgd'],
        "reg_knn":para_data['space_set']['reg']['knn']
}


try:
    if(reset_flag == "reset_default"):
        reset_parameters()
    if(reset_flag == "reset_settings"):
        json_s = os.path.join(os.path.dirname("./"), 'reset_settings.json')
        with open(json_s,'r') as d_file:
            para = json.load(d_file)
        json_s = os.path.join(os.path.dirname("./"), 'settings.json')
        w_file = open(json_s, "w",encoding='utf-8')
        w_file. truncate(0)
        json.dump(para, w_file)
        w_file.close()
    if(reset_flag == "no_confirm"):  
        reset_parameters()
        for i in custom_space.keys():
            if custom_space[i]!={}:
                model_type, algo_name=i.split('_')
                update_parameters(mode = model_type,estimator_name=algo_name,**custom_space[i])
except:
    print("Failed to Set Up the Searching Space, will Use the Default Settings!")


###------------------------ Settings PCTE Parameters--------------------------------------

json_path = os.path.join(os.path.dirname("./"), 'webapp.json')
with open(json_path, encoding='utf-8') as data_file:
    para_data = json.load(data_file)
data_file.close()

# Custom settings for the autoPP module
custom_pp = {}
custom_pp['encode_band'] = [int(para_data['autoPP']['encode_band'])]
custom_pp['scaler'] = para_data['autoPP']['scaler']
custom_pp['low_encode'] = para_data['autoPP']['low_encode']
custom_pp['high_encode'] = para_data['autoPP']['high_encode']
winsor_list = []
for i in para_data['autoPP']['winsorizer']:
    winsor_list.append((float(i),float(i)))
custom_pp['winsorizer'] = winsor_list
custom_pp['sparsity'] = [float(para_data['autoPP']['sparsity'])]
custom_pp['cols'] = [int(para_data['autoPP']['cols'])]

# Custom settings for the autoFS module
custom_fs = {
    "feature_num":int(para_data['autoFS']['feature_num']),
    "model_type_fs":para_data['autoFS']['model_type_fs'],
    "algo_fs":para_data['autoFS']['algo_fs']
}

# Custom settings for the autoCV module
custom_cv = {
        "model_type_cv":para_data['autoCV']['model_type_cv'],
        "method_cv":para_data['autoCV']['method_cv'],
        "algo_cv":para_data['autoCV']['algo_cv']
}


# Custom settings for input dataset
custom_input = {
    "filename": para_data['filename'],
    "label_col": para_data['label_col']
}

###------------------------ Run a PCTE Workflow--------------------------------------
df = pd.read_csv('./input/' + custom_input['filename'])

if custom_fs['model_type_fs'] == "cls" and custom_cv['model_type_cv'] == "cls":
    if custom_cv['method_cv'] == 'fastClassifier':
        # Create Pipeline Cluster Traversal Experiments by autoPipe
        pipe = autoPipe(
        [("autoPP",dynaPreprocessing(custom_parameters = custom_pp, label_col = custom_input['label_col'], model_type = "cls")),
        ("datasets_splitting",pipeline_splitting_rule(val_size = 0.2, test_size = 0.2, random_state = 13)),
        ("autoFS",dynaFS_clf(custom_selectors = custom_fs['algo_fs'],fs_num = custom_fs['feature_num'], random_state=13, cv = 5, in_pipeline = True, input_from_file = False)),
        ("autoCV",fastClassifier(custom_estimators = custom_cv['algo_cv'],random_state = 13,cv_num = 5,in_pipeline = True, input_from_file = False)),
        ("model_evaluate",evaluate_model(model_type = "cls"))])
    elif custom_cv['method_cv'] == 'dynaClassifier':
        # Create Pipeline Cluster Traversal Experiments by autoPipe
        pipe = autoPipe(
        [("autoPP",dynaPreprocessing(custom_parameters = custom_pp, label_col = custom_input['label_col'], model_type = "cls")),
        ("datasets_splitting",pipeline_splitting_rule(val_size = 0.2, test_size = 0.2, random_state = 13)),
        ("autoFS",dynaFS_clf(custom_selectors = custom_fs['algo_fs'],fs_num = custom_fs['feature_num'], random_state=13, cv = 5, in_pipeline = True, input_from_file = False)),
        ("autoCV",dynaClassifier(custom_estimators = custom_cv['algo_cv'],random_state = 13,cv_num = 5,in_pipeline = True, input_from_file = False)),
        ("model_evaluate",evaluate_model(model_type = "cls"))])
elif custom_fs['model_type_fs'] == "reg" and custom_cv['model_type_cv'] == "reg":
    if custom_cv['method_cv'] == 'fastRegressor':
        # Create Pipeline Cluster Traversal Experiments by autoPipe
        pipe = autoPipe(
        [("autoPP",dynaPreprocessing(custom_parameters = custom_pp, label_col = custom_input['label_col'], model_type = "reg")),
        ("datasets_splitting",pipeline_splitting_rule(val_size = 0.2, test_size = 0.2, random_state = 13)),
        ("autoFS",dynaFS_reg(custom_selectors = custom_fs['algo_fs'],fs_num = custom_fs['feature_num'], random_state=13, cv = 5, in_pipeline = True, input_from_file = False)),
        ("autoCV",fastRegressor(custom_estimators = custom_cv['algo_cv'],random_state = 13,cv_num = 5,in_pipeline = True, input_from_file = False)),
        ("model_evaluate",evaluate_model(model_type = "reg"))])
    elif custom_cv['method_cv'] == 'dynaRegressor':
        # Create Pipeline Cluster Traversal Experiments by autoPipe
        pipe = autoPipe(
        [("autoPP",dynaPreprocessing(custom_parameters = custom_pp, label_col = custom_input['label_col'], model_type = "reg")),
        ("datasets_splitting",pipeline_splitting_rule(val_size = 0.2, test_size = 0.2, random_state = 13)),
        ("autoFS",dynaFS_reg(custom_selectors = custom_fs['algo_fs'],fs_num = custom_fs['feature_num'], random_state=13, cv = 5, in_pipeline = True, input_from_file = False)),
        ("autoCV",dynaRegressor(custom_estimators = custom_cv['algo_cv'],random_state = 13,cv_num = 5,in_pipeline = True, input_from_file = False)),
        ("model_evaluate",evaluate_model(model_type = "reg"))])

DICT_PREPROCESSING,DICT_FEATURE_SELECTION,DICT_MODELS_EVALUATION,DICT_DATA,dyna_report= pipe.fit(df)

###------------------------ Save Outputs --------------------------------------
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
# Save the outputs as pickles for further analysis and visualization
save_obj(DICT_PREPROCESSING,"dict_preprocess")
save_obj(DICT_DATA,"dict_data")
save_obj(DICT_MODELS_EVALUATION,"dict_models_evaluate")
save_obj(dyna_report,"dyna_report")

###------------------------ Load PCTE Outputs --------------------------------------
# Load the outputs from picles
DICT_PREP = load_obj("dict_preprocess")
dyna_report = load_obj("dyna_report")
DICT_DATA = load_obj("dict_data")

###------------------------ PCTE Outputs Visualization --------------------------------------
import shutil
def move(src, dest):
    shutil.move(src, dest)

try:
    from optimalflow.autoViz import autoViz
    viz = autoViz(preprocess_dict=DICT_PREP,report=dyna_report)
    viz.clf_model_retrieval(metrics='accuracy')
    move('./Pipeline Cluster Retrieval Diagram.html','./templates/diagram.html')
    viz = autoViz(report = dyna_report)
    viz.clf_table_report()
    move('./Pipeline Cluster Model Evaluation Report.html','./templates/report.html')
except:
    try:
        viz = autoViz(report = dyna_report)
        viz.reg_table_report()
        move('./static/img/no-cls-output.html','./templates/diagram.html')
        move('./Pipeline Cluster Model Evaluation Report.html','./template/report.html')
    except:
        print('No Visualization Outputs found!')

print("PCTE Workflow's Done. More details of results are in LogsViewer & Visualization Page. Thanks for using OptimalFlow!! --Tony Dong")

input("\n\nPress the enter key to exit.")