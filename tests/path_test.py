import importlib.resources
import json
from dynapipe.utilis_func import export_parameters,reset_parameters,update_parameters

# import pandas as pd
# from pandas.io.json import json_normalize

# with open('parameters.json','r') as data_file:
#     data_json = json.load(data_file)
# flatten_data = pd.DataFrame(pd.json_normalize(data_json))
# flatten_data.head(3)


# Demo for parameter update, export, and reset:
update_parameters(mode = "cls", estimator_name = "svm", C=[0.1,0.2],kernel=["linear"])
export_parameters()
reset_parameters()

# data_file = open('./parameters.json','r')
# para_data = json.load(data_file)
# print(para_data["cls"]["lgr"])

# dict1 = para_data["cls"]["lgr"]
# print (dict1)
# dict1['random_state'] = 12
# print (dict1)
