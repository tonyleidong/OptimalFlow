#!/usr/bin/env python

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE,RFECV, f_regression, f_classif
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class clf_fs:
    def __init__(self,fs_num = None ,random_state = None,cv = None):
        self.fs_num = fs_num
        self.random_state = random_state
        self.cv = cv
    def kbest_f(self):
        selector = SelectKBest(score_func = f_classif, k = self.fs_num)
        return (selector)
    def kbest_chi2(self):
        selector = SelectKBest(score_func = chi2, k = self.fs_num)
        return (selector)
    def rfe_lr(self):
        estimator = LogisticRegression()
        selector = RFE(estimator, n_features_to_select = self.fs_num)
        return(selector)
    def rfe_svm(self):
        estimator = SVC(kernel="linear")
        selector = RFE(estimator, n_features_to_select = self.fs_num)
        return(selector)
    def rfe_tree(self):
        estimator = DecisionTreeClassifier()
        selector = RFE(estimator, n_features_to_select = self.fs_num)
        return(selector)
    def rfe_rf(self):
        estimator = RandomForestClassifier(max_depth = 3, n_estimators = 5)
        selector = RFE(estimator, n_features_to_select = self.fs_num)
        return(selector)
    def rfecv_svm(self):
        estimator = SVC(kernel="linear")
        selector = RFECV(estimator, min_features_to_select = self.fs_num, cv = self.cv)
        return(selector)
    def rfecv_tree(self):
        estimator = DecisionTreeClassifier()
        selector = RFECV(estimator, min_features_to_select = self.fs_num, cv = self.cv)
        return(selector)
    def rfecv_rf(self):
        estimator = RandomForestClassifier(max_depth = 3, n_estimators = 5)
        selector = RFECV(estimator, min_features_to_select = self.fs_num, cv = self.cv)
        return(selector)


class reg_fs:
    def __init__(self,fs_num,random_state = None,cv = None):
        self.fs_num = fs_num
        self.random_state = random_state
        self.cv = cv
    def kbest_f(self):
        selector = SelectKBest(score_func = f_regression, k = self.fs_num)
        return (selector)
    def rfe_svm(self):
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select = self.fs_num)
        return(selector)
    def rfe_tree(self):
        estimator = DecisionTreeRegressor()
        selector = RFE(estimator, n_features_to_select = self.fs_num)
        return(selector)
    def rfe_rf(self):
        estimator = RandomForestRegressor(max_depth = 3, n_estimators = 5)
        selector = RFE(estimator, n_features_to_select = self.fs_num)
        return(selector)
    def rfecv_svm(self):
        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, min_features_to_select = self.fs_num, cv = self.cv)
        return(selector)
    def rfecv_tree(self):
        estimator = DecisionTreeRegressor()
        selector = RFECV(estimator, min_features_to_select = self.fs_num, cv = self.cv)
        return(selector)
    def rfecv_rf(self):
        estimator = RandomForestRegressor(max_depth = 3, n_estimators = 5)
        selector = RFECV(estimator, min_features_to_select = self.fs_num, cv = self.cv)
        return(selector)


