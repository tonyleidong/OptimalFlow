#!/usr/bin/env python

import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LogisticRegression,LinearRegression,HuberRegressor,RidgeCV,RidgeClassifierCV,LassoCV,SGDRegressor,SGDClassifier
from sklearn.svm import SVC,SVR,LinearSVR,LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostRegressor,AdaBoostClassifier,HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
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
    def __init__(self,cv_val = None,random_state = None, fast_flag = False,n_comb = 10):
        self.cv = cv_val
        self.random_state = [random_state]
        self.fast_flag = fast_flag
        self.n_comb = n_comb
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
        if(self.fast_flag):
            return (RandomizedSearchCV(lgr_cv, parameters,cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(lgr_cv, parameters,cv = self.cv))
    
    def svm(self):
        warnings.warn = warn
        svm_cv = SVC()
        parameters = para_data["cls"]["svm"]
        # parameters = {
        #     'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        #     'C': [0.1, 1, 10]
        #     }
        if(self.fast_flag):
            return (RandomizedSearchCV(svm_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(svm_cv, parameters, cv = self.cv))
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

        if(self.fast_flag):
            return (RandomizedSearchCV(mlp_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(mlp_cv, parameters, cv = self.cv))
    
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
        if(self.fast_flag):
            return (RandomizedSearchCV(ada_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(ada_cv, parameters, cv = self.cv))    
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
        if(self.fast_flag):
            return (RandomizedSearchCV(rf_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(rf_cv, parameters, cv = self.cv)) 
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
        if(self.fast_flag):
            return (RandomizedSearchCV(gb_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(gb_cv, parameters, cv = self.cv)) 
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
        if(self.fast_flag):
            return (RandomizedSearchCV(xgb_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(xgb_cv, parameters, cv = self.cv)) 
    # New add on 8/10/2020
    def lsvc(self):
        warnings.warn = warn
        lsvc_cv = LinearSVC()
        parameters = para_data["cls"]["lsvc"]
        # parameters = {
        #     'C': [0.1, 1, 10]
        #     }
        if(self.fast_flag):
            return (RandomizedSearchCV(lsvc_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(lsvc_cv, parameters, cv = self.cv)) 
    def sgd(self):
        warnings.warn = warn
        sgd_cv = SGDClassifier()
        # parameters = {
        #     'shuffle': [True,False],
        #     'penalty': ['l2', 'l1', 'elasticnet'],
        #     'learning_rate': ['constant','optimal','invscaling']
        #     }
        parameters = para_data["cls"]["sgd"]
        if(self.fast_flag):
            return (RandomizedSearchCV(sgd_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(sgd_cv, parameters, cv = self.cv)) 
    def hgboost(self):
        warnings.warn = warn
        hgboost_cv = HistGradientBoostingClassifier()
        # parameters = {
        #     'max_depth': [3, 5, 7, 9],
        #     'learning_rate': [0.1, 0.2,0.3,0.4]
        #     }
        parameters = para_data["cls"]["hgboost"]
        if(self.fast_flag):
            return (RandomizedSearchCV(hgboost_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(hgboost_cv, parameters, cv = self.cv)) 

    def rgcv(self):
        warnings.warn = warn
        rgcv_cv = RidgeClassifierCV()
        # parameters = {
        #     'fit_intercept': [True,False]
        #     }
        parameters = para_data["cls"]["rgcv"]
        if(self.fast_flag):
            return (RandomizedSearchCV(rgcv_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(rgcv_cv, parameters, cv = self.cv)) 


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
    def __init__(self,cv_val = None,random_state = None, fast_flag = False,n_comb = 10):
        self.cv = cv_val
        self.random_state = [random_state]
        self.fast_flag = fast_flag
        self.n_comb = n_comb
        warnings.warn = warn

    def lr(self):
        warnings.warn = warn
        lr_cv = LinearRegression()
        # parameters = {
        #     'normalize' : [True,False]
        #     } 
        parameters = para_data["reg"]["lr"]
        if(self.fast_flag):
            return (RandomizedSearchCV(lr_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(lr_cv, parameters, cv = self.cv)) 
    def knn(self):
        warnings.warn = warn
        knn_cv = KNeighborsRegressor()
        # parameters = {
        #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        #     'n_neighbors': [5, 10, 15, 20, 25],
        #     'weights': ['uniform', 'distance'],
        #     }
        parameters = para_data["reg"]["knn"]
        if(self.fast_flag):
            return (RandomizedSearchCV(knn_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(knn_cv, parameters, cv = self.cv)) 
    def svm(self):
        warnings.warn = warn
        svm_cv = SVR()
        # parameters = {
        #     'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        #     'C': [0.1, 1, 10]
        #     }
        parameters = para_data["reg"]["svm"]
        if(self.fast_flag):
            return (RandomizedSearchCV(svm_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(svm_cv, parameters, cv = self.cv)) 
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
        if(self.fast_flag):
            return (RandomizedSearchCV(mlp_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(mlp_cv, parameters, cv = self.cv))   
    def rf(self):
        warnings.warn = warn
        rf_cv = RandomForestRegressor()
        # parameters = {
        #     'n_estimators': [5, 50, 250],
        #     'max_depth': [2, 4, 8, 16, 32]
        #     }
        parameters = para_data["reg"]["rf"]   
        if(self.fast_flag):
            return (RandomizedSearchCV(rf_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(rf_cv, parameters, cv = self.cv)) 
    def gb(self):
        warnings.warn = warn
        gb_cv = GradientBoostingRegressor()
        # parameters = {
        #     'n_estimators': [50,100,150,200,250,300],
        #     'max_depth': [3, 5, 7, 9],
        #     'learning_rate': [0.01, 0.1, 0.2,0.3,0.4]
        #     }
        parameters = para_data["reg"]["gb"]
        if(self.fast_flag):
            return (RandomizedSearchCV(gb_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(gb_cv, parameters, cv = self.cv)) 
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
        if(self.fast_flag):
            return (RandomizedSearchCV(tree_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(tree_cv, parameters, cv = self.cv))
      
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
        if(self.fast_flag):
            return (RandomizedSearchCV(ada_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(ada_cv, parameters, cv = self.cv))
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
        if(self.fast_flag):
            return (RandomizedSearchCV(xgb_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(xgb_cv, parameters, cv = self.cv))
    # # New add August 5,2020

    def hgboost(self):
        warnings.warn = warn
        hgboost_cv = HistGradientBoostingRegressor()
        # parameters = {
        #     'max_depth': [3, 5, 7, 9],
        #     'learning_rate': [0.1, 0.2,0.3,0.4]
        #     }
        parameters = para_data["reg"]["hgboost"]
        if(self.fast_flag):
            return (RandomizedSearchCV(hgboost_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(hgboost_cv, parameters, cv = self.cv))

    def huber(self):
        warnings.warn = warn
        huber_cv = HuberRegressor()
        # parameters = {
        #     'fit_intercept' : [True,False]
        #     }
        parameters = para_data["reg"]["huber"]
        if(self.fast_flag):
            return (RandomizedSearchCV(huber_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(huber_cv, parameters, cv = self.cv))

    def rgcv(self):
        warnings.warn = warn
        rgcv_cv = RidgeCV()
        # parameters = {
        #     'fit_intercept': [True,False]
        #     }
        parameters = para_data["reg"]["rgcv"]
        if(self.fast_flag):
            return (RandomizedSearchCV(rgcv_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(rgcv_cv, parameters, cv = self.cv))

    def cvlasso(self):
        warnings.warn = warn
        cvlasso_cv = LassoCV()
        # parameters = {
        #     'fit_intercept': [True,False]
        #     }
        parameters = para_data["reg"]["cvlasso"]
        if(self.fast_flag):
            return (RandomizedSearchCV(cvlasso_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(cvlasso_cv, parameters, cv = self.cv))
    def sgd(self):
        warnings.warn = warn
        sgd_cv = SGDRegressor()
        # parameters = {
        #     'shuffle': [True,False],
        #     'penalty': ['l2', 'l1', 'elasticnet'],
        #     'learning_rate': ['constant','optimal','invscaling']
        #     }
        parameters = para_data["reg"]["sgd"]
        if(self.fast_flag):
            return (RandomizedSearchCV(sgd_cv, parameters, cv = self.cv, n_iter = self.n_comb))
        else:
            return (GridSearchCV(sgd_cv, parameters, cv = self.cv))