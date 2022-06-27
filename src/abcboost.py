import os
from ctypes import * 
import numpy as np

def train(train_Y,train_X,model_name = "abcrobustlogit"):
  libc = cdll.LoadLibrary(os.getcwd() + '/libabcboost.so')
  train_Y = np.array(train_Y).astype(float)
  train_X = np.array(train_X).astype(float)
  n_row_y = len(train_Y)
  n_row_x = len(train_X)
  assert n_row_x > 0, '[ERROR] No data found in train_X.'
  assert n_row_x == n_row_y, '[ERROR] Rows in train_Y and train_X do not match.'
  n_col_x = len(train_X[0])
  train_Y = train_Y.flatten()
  train_X = np.transpose(train_X).flatten()
  model = libc.train(ctypes.c_void_p(train_Y.ctypes.data),ctypes.c_void_p(train_X.ctypes.data),ctypes.c_int(n_row_x),ctypes.c_int(n_row_y),ctypes.c_char_p(model_name))


