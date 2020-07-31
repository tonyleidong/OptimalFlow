#!/usr/bin/env python

import math
import pandas as pd
import numpy as np
from numpy import array
from numpy import count_nonzero
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

class PPtools:
  def __init__(self, data = None, label_col = None,model_type = 'reg'):
    self.snapshots = {}
    self.log = []
    self.high_cardinal_cols = []
    self.configure(data = data, label_col = label_col)
    self.model_type = model_type
    self.sparsity = 0

  def configure(self, data = None, label_col = None):
    if (type(label_col) is str or type(label_col) is int):
      self.label_col = label_col
    if (type(data) is str):
      self.initial_data = pd.read_csv(data)
      self.data = self.initial_data.copy()
      self.log = []
    if (type(data) is pd.DataFrame):
      self.initial_data = data
      self.data = self.initial_data.copy()
      self.log = []
    if(self.data[self.label_col].dtypes!= np.float64 and self.data[self.label_col].dtypes!= np.int64):
      self.data[self.label_col] = self.data[self.label_col].fillna("NaN")
      self.data[self.label_col] = pd.DataFrame(LabelEncoder().fit_transform(self.data[self.label_col]))

  def split_category_cols(self):
    non_label_list = self.data.columns.difference([self.label_col])
    self.non_label_data = self.data[non_label_list]
    self.cat_df = self.non_label_data.select_dtypes(exclude=['number'])
    self.num_df = self.non_label_data.select_dtypes(include = 'number')
    self._log("PPtools.split_category_cols(): Split input to category df and numeric df.")

  def remove_feature(self, feature_name):
    del self.data[feature_name]
    self._log("PPtools.remove_feature('{0}')".format(feature_name))

  def extract_feature(self, old_featre, new_feature, mapper = None):
    new_feature_column = map(mapper, self.data[old_featre])
    self.data[new_feature] = list(new_feature_column)
    self._log("PPtools.extract_feature({0}, {1}, {2})".format(old_featre, new_feature, mapper))

  def impute_tool(self):
    column_names = self.num_df.columns
    imp = SimpleImputer()
    imp.fit(self.num_df[column_names])
    self.num_df[column_names] = imp.transform(self.num_df[column_names])
    self.cat_df = self.cat_df.fillna("NaN")
    self._log("PPtools.impute_tool()")

  def scale_tool(self,df = None,sc_type = None):
    if sc_type == "None":
      self._log("PPtools.scale_tool() - None")
      return(df)

    if sc_type == "standard":
      self._log("PPtools.scale_tool() - StandardScaler")
      return(pd.DataFrame(preprocessing.StandardScaler().fit_transform(df),columns = df.columns))

    if sc_type == "minmax":
      self._log("PPtools.scale_tool() - MinMaxScaler")
      return(pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df),columns = df.columns))

    if sc_type == "maxabs":
      self._log("PPtools.scale_tool() - MaxAbsScaler")
      return(pd.DataFrame(preprocessing.MaxAbsScaler().fit_transform(df),columns = df.columns))

    if sc_type == "robust":
      self._log("PPtools.scale_tool() - RobustScaler")
      return(pd.DataFrame(preprocessing.RobustScaler().fit_transform(df),columns = df.columns))

  def winzorize_tool(self, lower_ban = None,upper_ban = None):
    for i in list(self.num_df.columns):
      self.num_df[i] = winsorize(self.num_df[i], limits=[lower_ban,upper_ban])
      
  def remove_zero_col_tool(self, data = None):
    return(data.loc[:, (data!= 0).any(axis=0)])

  def encode_tool(self,en_type = None ,category_col = None):
    if en_type == 'onehot':
      if(self.model_type == 'reg'):
        temp_df = pd.get_dummies(self.cat_df,prefix = ['onehot_'+category_col],columns = [category_col], drop_first = True)
        return(temp_df.loc[:, temp_df.columns.str.startswith('onehot_')])
      elif(self.model_type == 'cls'):
        temp_df = pd.get_dummies(self.cat_df,prefix = ['onehot_'+category_col],columns = [category_col], drop_first = False)
        return(temp_df.loc[:, temp_df.columns.str.startswith('onehot_')])
    if en_type == 'label':
      return(pd.DataFrame(LabelEncoder().fit_transform(self.cat_df[category_col]),columns=['Label_'+category_col]))
    
    if en_type == 'frequency':
      fe = self.cat_df.groupby(category_col).size()/len(self.cat_df)
      
      return(self.cat_df[category_col].map(fe).to_frame().rename(columns = {category_col:'Frequency_'+category_col}))

    if en_type == 'mean':
      mean_encoder = self.data.groupby(category_col)[self.label_col].mean()
      self.cat_df["Mean_"+category_col] = self.cat_df[category_col].map(mean_encoder)
      return(pd.DataFrame(self.cat_df["Mean_"+category_col]))


  def sparsity_tool(self, data = None):
    return(1.0 - (count_nonzero(data)/float(data.size)))

  def use_snapshot(self, name):
    self.data = self.snapshots[name]["data"]
    self.log = self.snapshots[name]["log"]

  def _log(self, string):
    self.log.append(string)