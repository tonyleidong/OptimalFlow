import pickle
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Load the outputs from picles
DICT_PREP = load_obj("dict_preprocess")
dyna_report = load_obj("dyna_report")
DICT_DATA = load_obj("dict_data")

import shutil
def move(src, dest):
    shutil.move(src, dest)

try:
    from optimalflow.autoViz import autoViz
    viz = autoViz(preprocess_dict=DICT_PREP,report=dyna_report)
    viz.clf_model_retrieval(metrics='accuracy')
    # move('./Pipeline Cluster Retrieval Diagram.html','./templates/diagram.html')
    viz = autoViz(report = dyna_report)
    viz.clf_table_report()
    # move('./Pipeline Cluster Model Evaluation Report.html','./templates/report.html')
except:
    try:
        viz = autoViz(report = dyna_report)
        viz.reg_table_report()
        move('./static/img/no-cls-output.html','./templates/diagram.html')
        move('./Pipeline Cluster Model Evaluation Report.html','./template/report.html')
    except:
        print('No Visualization Outputs found!')


input("\n\nPress the enter key to exit.")