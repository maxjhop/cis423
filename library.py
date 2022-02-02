import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score#, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split



class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    if self.action == "drop":
      X_ = X.drop(columns=self.column_list)

    else:
      X_ = X.drop(columns=(set(X.columns.to_list())-set(self.column_list)))
    
    return X_
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
#This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column   #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
    
  
  #fill in the rest below
  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    X_ = pd.get_dummies(X,
                                prefix=self.target_column,    #your choice
                                prefix_sep='_',     #your choice
                                columns=[self.target_column],
                                dummy_na=self.dummy_na,    #will try to impute later so leave NaNs in place
                                drop_first=self.drop_first    #will drop Belfast and infer it
                                )
    
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(df)} instead.'
    assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])

    #compute mean of column - look for method
    m = X[self.target_column].mean()
    #compute std of column - look for method
    sigma = X[self.target_column].std()
    minb, maxb = ((m - 3 * sigma), (m + 3 * sigma))#(lower bound, upper bound)
    X['Fare'] = X['Fare'].clip(lower=minb, upper=maxb)
    return X

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence
  
  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    q1 = X_[self.target_column].quantile(0.25)
    q3 = X_[self.target_column].quantile(0.75)
    iqr = q3-q1
    outer_low = q1-3*iqr
    outer_high = q3+3*iqr
    inner_low = q1 - 1.5*iqr
    inner_high = q3 + 1.5*iqr
    if self.fence == 'inner':
      X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)
    elif self.fence == 'outer':
      X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
  #fill in rest below
  def fit(self, X, y = None):
    print("not implemented")
    return X

  def transform(self, mtx):
    new_df = mtx.copy()
    for column in new_df:
      mi = new_df[column].min()
      mx = new_df[column].max()
      denominator = mx - mi
      new_df[column] -= mi #x - min(x) <- top of function
      new_df[column] /= denominator
    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform", add_indicator=False):
    self.n_neighbors = n_neighbors
    self.weights=weights 
    self.add_indicator=add_indicator

  #fill in rest below
  def fit(self, X, y = None):
    print("Warning: KNNTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    columns = X_.columns.tolist()
    imputer = KNNImputer(n_neighbors = self.n_neighbors, weights = self.weights, add_indicator = self.add_indicator)  
    imputed_data = imputer.fit_transform(X_)
    X_ = pd.DataFrame(imputed_data, columns=columns)
    return X_
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
def find_random_state(df, labels, n=200):
  var = []  #collect test_error/train_error where error based on F1 score

  #2 minutes
  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)  #predict against training set
      test_pred = model.predict(test_X)    #predict against test set
      train_error = f1_score(train_y, train_pred)  #how bad did we do with prediction on training data?
      test_error = f1_score(test_y, test_pred)     #how bad did we do with prediction on test data?
      error_ratio = test_error/train_error        #take the ratio
      var.append(error_ratio)

  rs_value = sum(var)/len(var)
  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx
