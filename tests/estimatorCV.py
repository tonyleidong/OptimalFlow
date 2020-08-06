#!/usr/bin/env python

import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression,HuberRegressor,RidgeCV,BayesianRidge,LassoCV,ElasticNetCV,SGDRegressor
from sklearn.svm import SVC,SVR,NuSVR,LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,RandomForestClassifier,RandomForestRegressor,AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import json
import os

json_path = os.path.join(os.path.dirname(__file__), 'parameters.json')
with open(json_path, encoding='utf-8') as data_file:
    para_data = json.load(data_file)
data_file.close()

def warn(*args, **kwargs):
    pass

class clf_cv:
    """This class stores classification estimators.
    
    Parameters
    ----------
    random_state : int, default = None
        Random state value.
    
    cv_val : int, default = None
        # of folds for cross-validation.
    Example
    -------
    
    .. [Example]
    
    References
    ----------
    None
    """
    def __init__(self,cv_val = None,random_state = None):
        self.cv = cv_val
        self.random_state = [random_state]
        warnings.warn = warn

    def lgr(self):
        warnings.warn = warn
        lgr_cv = LogisticRegression()
        # parameters = {
        #     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        #     'random_state': [self.random_state]
        #     }
        parameters = para_data["cls"]["lgr"]
        parameters['random_state'] = self.random_state
        return (GridSearchCV(lgr_cv, parameters,cv = self.cv))
    def svm(self):
        warnings.warn = warn
        svm_cv = SVC()
        parameters = para_data["cls"]["svm"]
        # parameters = {
        #     'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        #     'C': [0.1, 1, 10]
        #     }
        return(GridSearchCV(svm_cv, parameters, cv = self.cv))
    def mlp(self):
        warnings.warn = warn
        mlp_cv = MLPClassifier()
        # parameters = {
        #     'hidden_layer_sizes': [(10,), (50,), (100,)],
        #     'activation': ['identity','relu', 'tanh', 'logistic'],
        #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
        #     'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        #     'solver' : ['lbfgs', 'sgd', 'adam'],
        #     'random_state': [self.random_state]
        # }
        tupled = []
        ori = para_data["cls"]["mlp"]['hidden_layer_sizes']
        if (not isinstance(ori[0], tuple)): 
            for i in ori:
                li = tuple([i,])
                tupled.append(li)
            para_data["cls"]["mlp"]['hidden_layer_sizes'] = tupled
        parameters = para_data["cls"]["mlp"]
        parameters['random_state'] = self.random_state
        return(GridSearchCV(mlp_cv, parameters, cv = self.cv))
    def ada(self):
        warnings.warn = warn
        ada_cv = AdaBoostClassifier()
        # parameters = {
        #     'n_estimators': [50,100,150],
        #     'learning_rate': [0.01,0.1, 1, 5, 10],
        #     'random_state': [self.random_state]  
        #     }
        parameters = para_data["cls"]["ada"]
        parameters['random_state'] = self.random_state
        return(GridSearchCV(ada_cv, parameters, cv=self.cv))
    def rf(self):
        warnings.warn = warn
        rf_cv = RandomForestClassifier()
        # parameters = {
        #     'n_estimators': [5, 50, 250],
        #     'max_depth': [2, 4, 8, 16, 32],
        #     'random_state': [self.random_state]
            
        #     }
        parameters = para_data["cls"]["rf"]
        parameters['random_state'] = self.random_state
        return(GridSearchCV(rf_cv, parameters, cv=self.cv))
    def gb(self):
        warnings.warn = warn
        gb_cv = GradientBoostingClassifier()
        # parameters = {
        #     'n_estimators': [50,100,150,200,250,300],
        #     'max_depth': [1, 3, 5, 7, 9],
        #     'learning_rate': [0.01, 0.1, 1, 10, 100],
        #     'random_state': [self.random_state]
        #     }
        parameters = para_data["cls"]["gb"]
        parameters['random_state'] = self.random_state
        return(GridSearchCV(gb_cv, parameters,cv=self.cv))
    def xgb(self):
        warnings.warn = warn
        xgb_cv = xgb.XGBClassifier()
        # parameters = {
        #     'n_estimators': [50,100,150,200,250,300],
        #     'max_depth': [3, 5, 7, 9],
        #     'learning_rate': [0.01, 0.1, 0.2,0.3,0.4],
        #     'verbosity' : [0]
        #     }
        parameters = para_data["cls"]["gb"]
        return(GridSearchCV(xgb_cv, parameters,cv=self.cv))

class reg_cv:
    """This class stores regression estimators.
    
    Parameters
    ----------
    random_state : int, default = None
        Random state value.
    
    cv_val : int, default = None
        # of folds for cross-validation.
    Example
    -------
    
    .. [Example]
    
    References
    ----------
    None
    """
    def __init__(self,cv_val = None,random_state = None):
        self.cv = cv_val
        self.random_state = [random_state]
        warnings.warn = warn

    def lr(self):
        warnings.warn = warn
        lr_cv = LinearRegression()
        parameters = {
            'normalize' : [True,False]
            } 
        return (GridSearchCV(lr_cv, parameters,cv = self.cv))
    def knn(self):
        warnings.warn = warn
        knn_cv = KNeighborsRegressor()
        # parameters = {
        #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        #     'n_neighbors': [5, 10, 15, 20, 25],
        #     'weights': ['uniform', 'distance'],
        #     }
        parameters = para_data["reg"]["knn"]
        return(GridSearchCV(knn_cv, parameters, cv = self.cv))
    def svm(self):
        warnings.warn = warn
        svm_cv = SVR()
        # parameters = {
        #     'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        #     'C': [0.1, 1, 10]
        #     }
        parameters = para_data["reg"]["svm"]
        return(GridSearchCV(svm_cv, parameters, cv = self.cv))
    def mlp(self):
        warnings.warn = warn
        mlp_cv = MLPRegressor()
        # parameters = {
        #     'hidden_layer_sizes': [(10,), (50,), (100,)],
        #     'activation': ['identity','relu', 'tanh', 'logistic'],
        #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
        #     'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        #     'solver' : ['lbfgs', 'adam'],
        #     'random_state': [self.random_state]
        # }
        tupled = []
        ori = para_data["reg"]["mlp"]['hidden_layer_sizes']
        if (not isinstance(ori[0], tuple)): 
            for i in ori:
                li = tuple([i,])
                tupled.append(li)
            para_data["reg"]["mlp"]['hidden_layer_sizes'] = tupled
        parameters = para_data["reg"]["mlp"]
        parameters['random_state'] = self.random_state
        return(GridSearchCV(mlp_cv, parameters, cv = self.cv))  
    def rf(self):
        warnings.warn = warn
        rf_cv = RandomForestRegressor()
        # parameters = {
        #     'n_estimators': [5, 50, 250],
        #     'max_depth': [2, 4, 8, 16, 32]
        #     }
        parameters = para_data["reg"]["rf"]   
        return(GridSearchCV(rf_cv, parameters, cv=self.cv))
    def gb(self):
        warnings.warn = warn
        gb_cv = GradientBoostingRegressor()
        # parameters = {
        #     'n_estimators': [50,100,150,200,250,300],
        #     'max_depth': [3, 5, 7, 9],
        #     'learning_rate': [0.01, 0.1, 0.2,0.3,0.4]
        #     }
        parameters = para_data["reg"]["gb"]
        return(GridSearchCV(gb_cv, parameters,cv=self.cv))
    def tree(self):
        warnings.warn = warn
        tree_cv = DecisionTreeRegressor()
        # parameters = {
        #     'splitter':['best', 'random'],
        #     'max_depth': [1, 3, 5, 7, 9],
        #     'random_state': [self.random_state],
        #     'min_samples_leaf':[1,3,5]
        #     }
        parameters = para_data["reg"]["tree"]
        return(GridSearchCV(tree_cv, parameters,cv=self.cv))
      
    def ada(self):
        warnings.warn = warn
        ada_cv = AdaBoostRegressor()
        # parameters = {
        #     'n_estimators': [50,100,150,200,250,300],
        #     'loss':['linear','square','exponential'],
        #     'learning_rate': [0.01, 0.1, 0.2,0.3,0.4],
        #     'random_state': [self.random_state]            
        #     }
        parameters = para_data["reg"]["ada"]
        parameters['random_state'] = self.random_state
        return(GridSearchCV(ada_cv, parameters,cv=self.cv))
    def xgb(self):
        warnings.warn = warn
        xgb_cv = xgb.XGBRegressor()
        # parameters = {
        #     'n_estimators': [50,100,150,200,250,300],
        #     'max_depth': [3, 5, 7, 9],
        #     'learning_rate': [0.01, 0.1, 0.2,0.3,0.4],
        #     'verbosity' : [0]
        #     }
        parameters = para_data["reg"]["xgb"]
        return(GridSearchCV(xgb_cv, parameters,cv=self.cv))
    # # New add August 5,2020
    # def extratrees(self):
    #     warnings.warn = warn
    #     extratrees_cv = ExtraTreesRegressor()
    #     parameters = {
    #         'n_estimators': [50,100,150,200,250,300],
    #         'max_depth': [1, 3, 5, 7, 9],
    #         'random_state': [self.random_state],
    #         'min_samples_leaf':[1,3,5]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(extratrees_cv, parameters,cv=self.cv)) 
    # def nsvr(self):
    #     warnings.warn = warn
    #     nusvr_cv = NuSVR()
    #     parameters = {
    #         'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    #         'C': [0.1, 1, 10],
    #         'nu': [0.3,0.5,1]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(nusvr_cv, parameters,cv=self.cv)) 

    # def hgboost(self):
    #     warnings.warn = warn
    #     hgboost_cv = HistGradientBoostingRegressor()
    #     parameters = {
    #         'max_depth': [3, 5, 7, 9],
    #         'learning_rate': [0.1, 0.2,0.3,0.4]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(hgboost_cv, parameters,cv=self.cv)) 

    # def bagging(self):
    #     warnings.warn = warn
    #     bagging_cv = BaggingRegressor()
    #     parameters = {
    #         'n_estimators': [50,100,150,200,250,300],
    #         'bootstrap' : [True,False],
    #         'bootstrap_features' : [True,False],
    #         'oob_score' : [True,False]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(bagging_cv, parameters,cv=self.cv)) 

    # def huber(self):
    #     warnings.warn = warn
    #     huber_cv = HuberRegressor()
    #     parameters = {
    #         'fit_intercept' : [True,False]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(huber_cv, parameters,cv=self.cv))

    # def lsvr(self):
    #     warnings.warn = warn
    #     lsvr_cv = LinearSVR()
    #     parameters = {
    #         'C': [1, 5, 10],
    #         'random_state': [self.random_state] 
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(lsvr_cv, parameters,cv=self.cv))

    # def rgcv(self):
    #     warnings.warn = warn
    #     rgcv_cv = RidgeCV()
    #     parameters = {
    #         'fit_intercept': [True,False]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(rgcv_cv, parameters,cv=self.cv))

    # def bayesrg(self):
    #     warnings.warn = warn
    #     bayesrg_cv = BayesianRidge()
    #     parameters = {
    #         'fit_intercept': [True,False]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(rgcv_cv, parameters,cv=self.cv))

    # def cvlasso(self):
    #     warnings.warn = warn
    #     cvlasso_cv = LassoCV()
    #     parameters = {
    #         'fit_intercept': [True,False]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(cvlasso_cv, parameters,cv=self.cv))

    # def elastic(self):
    #     warnings.warn = warn
    #     elastic_cv = ElasticNetCV()
    #     parameters = {
    #         'fit_intercept': [True,False]
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(cvlasso_cv, parameters,cv=self.cv))

    # def sgd(self):
    #     warnings.warn = warn
    #     sgd_cv = SGDRegressor()
    #     parameters = {
    #         'shuffle': [True,False],
    #         'penalty': ['l2', 'l1', 'elasticnet'],
    #         'learning_rate': ['constant','optimal','invscaling']
    #         }
    #     # parameters = para_data["reg"]["tree"]
    #     return(GridSearchCV(sgd_cv, parameters,cv=self.cv))