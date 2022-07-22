# Copyright 2022 The ABCBoost Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


abcboost_train <- function(train_Y,train_X,model_name,iter,leaves,shinkage,params=NULL){
  train_Y = as.numeric(train_Y)
  if(is(train_X,'sparseMatrix') == FALSE){
    train_X = as.matrix(train_X)
    if(is.matrix(train_X) == FALSE){
      stop("train_X must be a 2 dimensional matrix.")
    }
  }
	n_row_y <- length(train_Y)
	n_row_x <- nrow(train_X)
	n_col_x <- ncol(train_X)
	if(n_row_y != n_row_x){
		msg <- sprintf('Rows in train_Y (%i) and train_X (%i) do not match.',n_row_y,n_row_x)
		stop(msg)
	}
  if(is(train_X, 'sparseMatrix') == TRUE){
    train_X <- t(as(train_X,'dgCMatrix'))
    leni = length(train_X@i)
    lenp = length(train_X@p)
    model <-.Call("train_sparse",as.double(train_Y),as.integer(train_X@i),as.integer(leni),as.integer(train_X@p),as.integer(lenp),as.double(train_X@x),as.integer(n_row_x),as.integer(n_col_x),model_name,as.integer(iter),as.integer(leaves),as.double(shinkage),as.list(params))
  }else{
    model <-.Call("train",as.double(train_Y),as.double(train_X),as.integer(n_row_x),as.integer(n_col_x),model_name,as.integer(iter),as.integer(leaves),as.double(shinkage),as.list(params))
  }
	return(model)
}

abcboost_test <- function(test_Y,test_X,model,params=NULL){
  test_Y = as.numeric(test_Y)
  if(is(test_X,'sparseMatrix') == FALSE){
    test_X = as.matrix(test_X)
    if(is.matrix(test_X) == FALSE){
      stop("test_X must be a 2 dimensional matrix.")
    }
  }
	n_row_y <- length(test_Y)
	n_row_x <- nrow(test_X)
	n_col_x <- ncol(test_X)
	if(n_row_y != n_row_x){
		msg <- sprintf('Rows in test_Y (%i) and test_X (%i) do not match.',n_row_y,n_row_x)
		stop(msg)
	}
  if(is(test_X, 'sparseMatrix') == TRUE){
    test_X <- t(as(test_X,'dgCMatrix'))
    leni = length(test_X@i)
    lenp = length(test_X@p)
	  ret <- .Call("test_sparse",as.double(test_Y),as.integer(test_X@i),as.integer(leni),as.integer(test_X@p),as.integer(lenp),as.double(test_X@x),as.integer(n_row_x),as.integer(n_col_x),model,as.list(params))
  }else{
	  ret <- .Call("test",as.double(test_Y),as.double(test_X),as.integer(n_row_x),as.integer(n_col_x),model,as.list(params))
  }
	return(ret)
}

abcboost_predict <- function(test_X,model,params=NULL){
  if(is(test_X,'sparseMatrix') == FALSE){
    test_X = as.matrix(test_X)
    if(is.matrix(test_X) == FALSE){
      stop("test_X must be a 2 dimensional matrix.")
    }
  }
	n_row_x <- nrow(test_X)
	n_col_x <- ncol(test_X)
  if(is(test_X, 'sparseMatrix') == TRUE){
    test_X <- t(as(test_X,'dgCMatrix'))
    leni = length(test_X@i)
    lenp = length(test_X@p)
	  ret <- .Call("predict_sparse",as.integer(test_X@i),as.integer(leni),as.integer(test_X@p),as.integer(lenp),as.double(test_X@x),as.integer(n_row_x),as.integer(n_col_x),model,as.list(params))
  }else{
	  ret <- .Call("predict",as.double(test_X),as.integer(n_row_x),as.integer(n_col_x),model,as.list(params))
  }
	return(ret)
}

abcboost_save_model <- function(model,path){
	.Call("saveModel",model,path)
}

abcboost_load_model <- function(path){
	model <-.Call("loadModel",path)
	return(model)
}


abcboost_read_libsvm <- function(path){
  data <- .Call("read_libsvm",path)
  i <- data$is
  p <- data$ps
  x <- data$xs
  y <- data$y
  i <- as.integer(unlist(i))
  p <- as.integer(unlist(p))
  x <- as.numeric(unlist(x))
  X <- sparseMatrix(i = i,p = p,x = x,index1=FALSE)
  ret = list()
  Y <- as.numeric(unlist(y))
  ret$X <- t(X)
  ret$Y <- Y
  return(ret)
}

