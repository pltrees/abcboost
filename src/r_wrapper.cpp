// Copyright 2022 The ABCBoost Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "data.h"
#include "model.h"
#include "tree.h"
#include "utils.h"

#include <R.h>
#include <Rinternals.h>

extern "C" {
SEXP train(SEXP Y, SEXP X, SEXP row, SEXP col, SEXP r_model_name, SEXP r_iter, SEXP r_leaves, SEXP r_shrinkage, SEXP params = NILSXP) {
  SEXP x = PROTECT(coerceVector(X, REALSXP));
  SEXP y = PROTECT(coerceVector(Y, REALSXP));
  int n_row = *INTEGER(row);
  int n_col = *INTEGER(col);
  std::string model_name = std::string(CHAR(STRING_ELT(r_model_name, 0)));
  int n_iter = *INTEGER(r_iter);
  int n_leaves = *INTEGER(r_leaves);
  double shrinkage = *REAL(r_shrinkage);

  ABCBoost::Config* config = new ABCBoost::Config();
  config->model_mode = "train";
  config->from_wrapper = true;
  config->model_mapping_name = "tmp" + std::to_string((unsigned int)clock()) + config->mapping_suffix;
  config->model_name = model_name;
  config->model_n_iterations = n_iter;
  config->tree_max_n_leaves = n_leaves;
  config->model_shrinkage = shrinkage;

  config->model_pretrained_path = "";
  config->save_model = false;
  config->save_log = false;

  if (params != NILSXP){
    int nfields = length(params);
    SEXP names = getAttrib(params,R_NamesSymbol);
    for(int i = 0;i < nfields;++i){
      std::string name = std::string(CHAR(STRING_ELT(names,i)));
      SEXP field = VECTOR_ELT(params,i);
      std::string val;
      if(TYPEOF(field) == INTSXP){
        val = std::to_string(*INTEGER(field));
      }else if(TYPEOF(field) == REALSXP){
        val = std::to_string(*REAL(field));
      }else if(TYPEOF(field) == STRSXP){
        val = std::string(CHAR(STRING_ELT(field, 0)));
      }else{
        printf("[Error] The param has to be either scalar or string");
        return R_NilValue;
      }
      try{
        config->parse(name,val);
      }catch (...){
        error("The program stopped due to the above error(s).");
        return R_NilValue;
      }
    }
  }

  ABCBoost::Data* data = new ABCBoost::Data(config);

  ABCBoost::GradientBoosting* model;
  config->model_use_logit =
      (config->model_name.find("logit") != std::string::npos);

  if (config->model_name == "mart" || config->model_name == "robustlogit") {
    model = new ABCBoost::Mart(data, config);
  } else if (config->model_name == "abcmart" ||
             config->model_name == "abcrobustlogit") {
    model = new ABCBoost::ABCMart(data, config);
  } else if (config->model_name == "regression") {
    config->model_is_regression = true;
    model = new ABCBoost::Regression(data, config);
  } else if (config->model_name == "lambdamart" || config->model_name == "lambdarank") {
    config->model_is_regression = true;
		config->model_use_logit = true;
    model = new ABCBoost::LambdaMart(data, config);
  } else if (config->model_name == "gbrank") {
    config->model_is_regression = 1;
		config->model_use_logit = true;
    model = new ABCBoost::GBRank(data, config);
  } else {
    printf("[ERROR] Unsupported model name %s\n", config->model_name.c_str());
    return R_NilValue;
  }

  config->mem_Y_matrix = REAL(y);
  config->mem_X_matrix = REAL(x);
  config->mem_n_row = n_row;
  config->mem_n_col = n_col;

  data->loadData(true);
  model->init();
  model->loadModel();
  model->setupExperiment();

  model->train();

  SEXP ret = R_MakeExternalPtr(model, R_NilValue, R_NilValue);
  UNPROTECT(2);
  return ret;
}

SEXP train_sparse(SEXP Y, SEXP X_i, SEXP r_leni, SEXP X_p, SEXP r_lenp, SEXP X_x, SEXP row, SEXP col, SEXP r_model_name, SEXP r_iter, SEXP r_leaves, SEXP r_shrinkage, SEXP params = NILSXP) {
  SEXP xi = PROTECT(coerceVector(X_i,INTSXP));
  SEXP xp = PROTECT(coerceVector(X_p,INTSXP));
  SEXP xx = PROTECT(coerceVector(X_x,REALSXP));


  SEXP y = PROTECT(coerceVector(Y, REALSXP));
  int leni = *INTEGER(r_leni);
  int lenp = *INTEGER(r_lenp);
  
  int* pxi = INTEGER(xi);
  int* pxp = INTEGER(xp);
  double* pxx = REAL(xx);
  std::vector<std::vector<std::pair<int,double>>> kvs;
  for(int i = 0;i < lenp - 1;++i){
    std::vector<std::pair<int,double>> vec;
    for(int j = pxp[i];j < pxp[i + 1];++j){
      vec.push_back(std::make_pair(pxi[j] + 1,pxx[j]));
    }
    kvs.push_back(vec);
  }

  int n_row = *INTEGER(row);
  int n_col = *INTEGER(col);
  std::string model_name = std::string(CHAR(STRING_ELT(r_model_name, 0)));
  int n_iter = *INTEGER(r_iter);
  int n_leaves = *INTEGER(r_leaves);
  double shrinkage = *REAL(r_shrinkage);

  ABCBoost::Config* config = new ABCBoost::Config();
  config->model_mode = "train";
  config->from_wrapper = true;
  config->model_mapping_name = "tmp" + std::to_string((unsigned int)clock()) + config->mapping_suffix;
  config->model_name = model_name;
  config->model_n_iterations = n_iter;
  config->tree_max_n_leaves = n_leaves;
  config->model_shrinkage = shrinkage;

  config->model_pretrained_path = "";
  config->save_model = false;
  config->save_log = false;

  if (params != NILSXP){
    int nfields = length(params);
    SEXP names = getAttrib(params,R_NamesSymbol);
    for(int i = 0;i < nfields;++i){
      std::string name = std::string(CHAR(STRING_ELT(names,i)));
      SEXP field = VECTOR_ELT(params,i);
      std::string val;
      if(TYPEOF(field) == INTSXP){
        val = std::to_string(*INTEGER(field));
      }else if(TYPEOF(field) == REALSXP){
        val = std::to_string(*REAL(field));
      }else if(TYPEOF(field) == STRSXP){
        val = std::string(CHAR(STRING_ELT(field, 0)));
      }else{
        printf("[Error] The param has to be either scalar or string");
        return R_NilValue;
      }
      try{
        config->parse(name,val);
      }catch (...){
        error("The program stopped due to the above error(s).");
        return R_NilValue;
      }
    }
  }

  ABCBoost::Data* data = new ABCBoost::Data(config);

  ABCBoost::GradientBoosting* model;
  config->model_use_logit =
      (config->model_name.find("logit") != std::string::npos);

  if (config->model_name == "mart" || config->model_name == "robustlogit") {
    model = new ABCBoost::Mart(data, config);
  } else if (config->model_name == "abcmart" ||
             config->model_name == "abcrobustlogit") {
    model = new ABCBoost::ABCMart(data, config);
  } else if (config->model_name == "regression") {
    config->model_is_regression = true;
    model = new ABCBoost::Regression(data, config);
  } else if (config->model_name == "lambdamart" || config->model_name == "lambdarank") {
    config->model_is_regression = true;
		config->model_use_logit = true;
    model = new ABCBoost::LambdaMart(data, config);
  } else if (config->model_name == "gbrank") {
    config->model_is_regression = 1;
		config->model_use_logit = true;
    model = new ABCBoost::GBRank(data, config);
  } else {
    printf("[ERROR] Unsupported model name %s\n", config->model_name.c_str());
    return R_NilValue;
  }

  config->mem_is_sparse = true;
  config->mem_Y_matrix = REAL(y);
  config->mem_X_kv = kvs;
  config->mem_n_row = n_row;
  config->mem_n_col = n_col;

  data->loadData(true);
  model->init();
  model->loadModel();
  model->setupExperiment();

  model->train();

  SEXP ret = R_MakeExternalPtr(model, R_NilValue, R_NilValue);
  UNPROTECT(4);
  return ret;
}

SEXP test(SEXP Y, SEXP X, SEXP row, SEXP col, SEXP r_model, SEXP params = NILSXP) {
  SEXP x = PROTECT(coerceVector(X, REALSXP));
  SEXP y = PROTECT(coerceVector(Y, REALSXP));
  int n_row = *INTEGER(row);
  int n_col = *INTEGER(col);

  ABCBoost::GradientBoosting* model =
      reinterpret_cast<ABCBoost::GradientBoosting*>(R_ExternalPtrAddr(r_model));
  ABCBoost::Config* config = model->getConfig();
  if (params != NILSXP){
    int nfields = length(params);
    SEXP names = getAttrib(params,R_NamesSymbol);
    for(int i = 0;i < nfields;++i){
      std::string name = std::string(CHAR(STRING_ELT(names,i)));
      SEXP field = VECTOR_ELT(params,i);
      std::string val;
      if(TYPEOF(field) == INTSXP){
        val = std::to_string(*INTEGER(field));
      }else if(TYPEOF(field) == REALSXP){
        val = std::to_string(*REAL(field));
      }else if(TYPEOF(field) == STRSXP){
        val = std::string(CHAR(STRING_ELT(field, 0)));
      }else{
        printf("[Error] The param has to be either scalar or string");
        return R_NilValue;
      }
      try{
        config->parse(name,val);
      }catch (...){
        error("The program stopped due to the above error(s).");
        return R_NilValue;
      }
    }
  }
  config->model_mode = "test";

  config->mem_Y_matrix = REAL(y);
  config->mem_X_matrix = REAL(x);
  config->mem_n_row = n_row;
  config->mem_n_col = n_col;
  config->from_wrapper = true;
  config->save_log = false;

  model->getData()->loadData(true);
  model->init();
  model->setupExperiment();

  model->testlog.clear();
  model->test();
  int n_classes = model->getData()->data_header.n_classes;
	std::vector<double> prob(1);
  std::vector<double> prediction(n_row);
  if(config->save_prob){
    prob.resize(n_row * n_classes);
  }
  model->returnPrediction(prediction.data(),prob.data());
  SEXP r_pred = PROTECT(allocMatrix(REALSXP, n_row, 1));
  for (int i = 0; i < n_row; ++i)
    REAL(r_pred)[i] = prediction[i];
  
  int testlog_col = model->testlog.size() > 0 ? model->testlog[0].size() : 0;
  int testlog_row = model->testlog.size();
  SEXP r_testlog = PROTECT(allocMatrix(REALSXP, testlog_row, testlog_col));
  for(int i = 0;i < testlog_row;++i){
    for(int j = 0;j < model->testlog[i].size();++j){
      REAL(r_testlog)[j * testlog_row + i] = model->testlog[i][j];
    }
  }

  SEXP r_prob;
  if(config->save_prob){
    r_prob = PROTECT(allocMatrix(REALSXP, n_row, n_classes));
    for (int i = 0; i < n_row * n_classes; ++i)
      REAL(r_prob)[i] = prob[i];
  }else{
    r_prob = PROTECT(allocMatrix(REALSXP, 1, 1));
  }
  
  SEXP ret;
  if(config->save_prob){
    const char *names[] = {"prediction", "testlog", "probability", ""};
    ret = PROTECT(mkNamed(VECSXP, names));
    SET_VECTOR_ELT(ret, 0, r_pred);
    SET_VECTOR_ELT(ret, 1, r_testlog);
    SET_VECTOR_ELT(ret, 2, r_prob);
  }else{
    const char *names[] = {"prediction", "testlog", ""};
    ret = PROTECT(mkNamed(VECSXP, names));
    SET_VECTOR_ELT(ret, 0, r_pred);
    SET_VECTOR_ELT(ret, 1, r_testlog);
  }

  UNPROTECT(6);
  model->testlog.clear();
  return ret;
}

SEXP predict(SEXP X, SEXP row, SEXP col, SEXP r_model, SEXP params = NILSXP) {
  SEXP x = PROTECT(coerceVector(X, REALSXP));
  int n_row = *INTEGER(row);
  int n_col = *INTEGER(col);

  ABCBoost::GradientBoosting* model =
      reinterpret_cast<ABCBoost::GradientBoosting*>(R_ExternalPtrAddr(r_model));
  ABCBoost::Config* config = model->getConfig();
  if (params != NILSXP){
    int nfields = length(params);
    SEXP names = getAttrib(params,R_NamesSymbol);
    for(int i = 0;i < nfields;++i){
      std::string name = std::string(CHAR(STRING_ELT(names,i)));
      SEXP field = VECTOR_ELT(params,i);
      std::string val;
      if(TYPEOF(field) == INTSXP){
        val = std::to_string(*INTEGER(field));
      }else if(TYPEOF(field) == REALSXP){
        val = std::to_string(*REAL(field));
      }else if(TYPEOF(field) == STRSXP){
        val = std::string(CHAR(STRING_ELT(field, 0)));
      }else{
        printf("[Error] The param has to be either scalar or string");
        return R_NilValue;
      }
      try{
        config->parse(name,val);
      }catch (...){
        error("The program stopped due to the above error(s).");
        return R_NilValue;
      }
    }
  }
  config->model_mode = "test";

  config->mem_Y_matrix = NULL;
  config->mem_X_matrix = REAL(x);
  config->mem_n_row = n_row;
  config->mem_n_col = n_col;
  config->from_wrapper = true;
  config->save_log = false;
  
  bool prev_no_label = config->no_label;
  config->no_label = true;

  model->getData()->loadData(true);
  model->init();
  model->setupExperiment();

  model->test();
  int n_classes = model->getData()->data_header.n_classes;
	std::vector<double> prob(1);
  std::vector<double> prediction(n_row);
  if(config->save_prob){
    prob.resize(n_row * n_classes);
  }
  model->returnPrediction(prediction.data(),prob.data());
  SEXP r_pred = PROTECT(allocMatrix(REALSXP, n_row, 1));
  for (int i = 0; i < n_row; ++i)
    REAL(r_pred)[i] = prediction[i];
  
  SEXP r_prob;
  if(config->save_prob){
    r_prob = PROTECT(allocMatrix(REALSXP, n_row, n_classes));
    for (int i = 0; i < n_row * n_classes; ++i)
      REAL(r_prob)[i] = prob[i];
  }else{
    r_prob = PROTECT(allocMatrix(REALSXP, 1, 1));
  }
  
  SEXP ret;
  if(config->save_prob){
    const char *names[] = {"prediction", "probability", ""};
    ret = PROTECT(mkNamed(VECSXP, names));
    SET_VECTOR_ELT(ret, 0, r_pred);
    SET_VECTOR_ELT(ret, 1, r_prob);
  }else{
    const char *names[] = {"prediction", ""};
    ret = PROTECT(mkNamed(VECSXP, names));
    SET_VECTOR_ELT(ret, 0, r_pred);
  }

  UNPROTECT(4);
  config->no_label = prev_no_label;
  return ret;
}

SEXP test_sparse(SEXP Y,SEXP X_i, SEXP r_leni, SEXP X_p, SEXP r_lenp, SEXP X_x, SEXP row, SEXP col, SEXP r_model, SEXP params = NILSXP) {
  SEXP xi = PROTECT(coerceVector(X_i,INTSXP));
  SEXP xp = PROTECT(coerceVector(X_p,INTSXP));
  SEXP xx = PROTECT(coerceVector(X_x,REALSXP));
  SEXP y = PROTECT(coerceVector(Y, REALSXP));
  int leni = *INTEGER(r_leni);
  int lenp = *INTEGER(r_lenp);
  int* pxi = INTEGER(xi);
  int* pxp = INTEGER(xp);
  double* pxx = REAL(xx);
  std::vector<std::vector<std::pair<int,double>>> kvs;
  for(int i = 0;i < lenp - 1;++i){
    std::vector<std::pair<int,double>> vec;
    for(int j = pxp[i];j < pxp[i + 1];++j){
      vec.push_back(std::make_pair(pxi[j] + 1,pxx[j]));
    }
    kvs.push_back(vec);
  }

  int n_row = *INTEGER(row);
  int n_col = *INTEGER(col);

  ABCBoost::GradientBoosting* model =
      reinterpret_cast<ABCBoost::GradientBoosting*>(R_ExternalPtrAddr(r_model));
  ABCBoost::Config* config = model->getConfig();  if (params != NILSXP){
    int nfields = length(params);
    SEXP names = getAttrib(params,R_NamesSymbol);
    for(int i = 0;i < nfields;++i){
      std::string name = std::string(CHAR(STRING_ELT(names,i)));
      SEXP field = VECTOR_ELT(params,i);
      std::string val;
      if(TYPEOF(field) == INTSXP){
        val = std::to_string(*INTEGER(field));
      }else if(TYPEOF(field) == REALSXP){
        val = std::to_string(*REAL(field));
      }else if(TYPEOF(field) == STRSXP){
        val = std::string(CHAR(STRING_ELT(field, 0)));
      }else{
        printf("[Error] The param has to be either scalar or string");
        return R_NilValue;
      }
      try{
        config->parse(name,val);
      }catch (...){
        error("The program stopped due to the above error(s).");
        return R_NilValue;
      }
    }
  }
  config->model_mode = "test";

  config->mem_is_sparse = true;
  config->mem_Y_matrix = REAL(y);
  config->mem_X_kv = kvs;
  config->mem_n_row = n_row;
  config->mem_n_col = n_col;
  config->from_wrapper = true;
  config->save_log = false;

  model->getData()->loadData(true);
  model->init();
  model->setupExperiment();

  model->testlog.clear();
  model->test();
  int n_classes = model->getData()->data_header.n_classes;
	std::vector<double> prob(1);
  std::vector<double> prediction(n_row);
  if(config->save_prob){
    prob.resize(n_row * n_classes);
  }
  model->returnPrediction(prediction.data(),prob.data());
  SEXP r_pred = PROTECT(allocMatrix(REALSXP, n_row, 1));
  for (int i = 0; i < n_row; ++i)
    REAL(r_pred)[i] = prediction[i];
  
  int testlog_col = model->testlog.size() > 0 ? model->testlog[0].size() : 0;
  int testlog_row = model->testlog.size();
  SEXP r_testlog = PROTECT(allocMatrix(REALSXP, testlog_row, testlog_col));
  for(int i = 0;i < testlog_row;++i){
    for(int j = 0;j < model->testlog[i].size();++j){
      REAL(r_testlog)[j * testlog_row + i] = model->testlog[i][j];
    }
  }

  SEXP r_prob;
  if(config->save_prob){
    r_prob = PROTECT(allocMatrix(REALSXP, n_row, n_classes));
    for (int i = 0; i < n_row * n_classes; ++i)
      REAL(r_prob)[i] = prob[i];
  }else{
    r_prob = PROTECT(allocMatrix(REALSXP, 1, 1));
  }
  
  SEXP ret;
  if(config->save_prob){
    const char *names[] = {"prediction", "testlog", "probability", ""};
    ret = PROTECT(mkNamed(VECSXP, names));
    SET_VECTOR_ELT(ret, 0, r_pred);
    SET_VECTOR_ELT(ret, 1, r_testlog);
    SET_VECTOR_ELT(ret, 2, r_prob);
  }else{
    const char *names[] = {"prediction", "testlog", ""};
    ret = PROTECT(mkNamed(VECSXP, names));
    SET_VECTOR_ELT(ret, 0, r_pred);
    SET_VECTOR_ELT(ret, 1, r_testlog);
  }

  UNPROTECT(8);
  model->testlog.clear();
  return ret;
}

SEXP predict_sparse(SEXP X_i, SEXP r_leni, SEXP X_p, SEXP r_lenp, SEXP X_x, SEXP row, SEXP col, SEXP r_model, SEXP params = NILSXP) {
  SEXP xi = PROTECT(coerceVector(X_i,INTSXP));
  SEXP xp = PROTECT(coerceVector(X_p,INTSXP));
  SEXP xx = PROTECT(coerceVector(X_x,REALSXP));
  int leni = *INTEGER(r_leni);
  int lenp = *INTEGER(r_lenp);
  int* pxi = INTEGER(xi);
  int* pxp = INTEGER(xp);
  double* pxx = REAL(xx);
  std::vector<std::vector<std::pair<int,double>>> kvs;
  for(int i = 0;i < lenp - 1;++i){
    std::vector<std::pair<int,double>> vec;
    for(int j = pxp[i];j < pxp[i + 1];++j){
      vec.push_back(std::make_pair(pxi[j] + 1,pxx[j]));
    }
    kvs.push_back(vec);
  }

  int n_row = *INTEGER(row);
  int n_col = *INTEGER(col);

  ABCBoost::GradientBoosting* model =
      reinterpret_cast<ABCBoost::GradientBoosting*>(R_ExternalPtrAddr(r_model));
  ABCBoost::Config* config = model->getConfig();
  if (params != NILSXP){
    int nfields = length(params);
    SEXP names = getAttrib(params,R_NamesSymbol);
    for(int i = 0;i < nfields;++i){
      std::string name = std::string(CHAR(STRING_ELT(names,i)));
      SEXP field = VECTOR_ELT(params,i);
      std::string val;
      if(TYPEOF(field) == INTSXP){
        val = std::to_string(*INTEGER(field));
      }else if(TYPEOF(field) == REALSXP){
        val = std::to_string(*REAL(field));
      }else if(TYPEOF(field) == STRSXP){
        val = std::string(CHAR(STRING_ELT(field, 0)));
      }else{
        printf("[Error] The param has to be either scalar or string");
        return R_NilValue;
      }
      try{
        config->parse(name,val);
      }catch (...){
        error("The program stopped due to the above error(s).");
        return R_NilValue;
      }
    }
  }
  config->model_mode = "test";

  config->mem_is_sparse = true;
  config->mem_Y_matrix = NULL;
  config->mem_X_kv = kvs;
  config->mem_n_row = n_row;
  config->mem_n_col = n_col;
  config->from_wrapper = true;
  config->save_log = false;
  
  bool prev_no_label = config->no_label;
  config->no_label = true;

  model->getData()->loadData(true);
  model->init();
  model->setupExperiment();

  model->test();
  int n_classes = model->getData()->data_header.n_classes;
	std::vector<double> prob(1);
  std::vector<double> prediction(n_row);
  if(config->save_prob){
    prob.resize(n_row * n_classes);
  }
  model->returnPrediction(prediction.data(),prob.data());
  SEXP r_pred = PROTECT(allocMatrix(REALSXP, n_row, 1));
  for (int i = 0; i < n_row; ++i)
    REAL(r_pred)[i] = prediction[i];
  
  SEXP r_prob;
  if(config->save_prob){
    r_prob = PROTECT(allocMatrix(REALSXP, n_row, n_classes));
    for (int i = 0; i < n_row * n_classes; ++i)
      REAL(r_prob)[i] = prob[i];
  }else{
    r_prob = PROTECT(allocMatrix(REALSXP, 1, 1));
  }
  
  SEXP ret;
  if(config->save_prob){
    const char *names[] = {"prediction", "probability", ""};
    ret = PROTECT(mkNamed(VECSXP, names));
    SET_VECTOR_ELT(ret, 0, r_pred);
    SET_VECTOR_ELT(ret, 1, r_prob);
  }else{
    const char *names[] = {"prediction", ""};
    ret = PROTECT(mkNamed(VECSXP, names));
    SET_VECTOR_ELT(ret, 0, r_pred);
  }

  UNPROTECT(6);
  config->no_label = prev_no_label;
  return ret;
}

SEXP saveModel(SEXP r_model, SEXP r_path) {
  ABCBoost::GradientBoosting* model =
      reinterpret_cast<ABCBoost::GradientBoosting*>(R_ExternalPtrAddr(r_model));
  std::string path = std::string(CHAR(STRING_ELT(r_path, 0)));
  model->setExperimentPath(path);
  model->getConfig()->from_wrapper = true;
  model->getConfig()->model_suffix = "";

  model->saveModel(model->getConfig()->model_n_iterations);

  return R_NilValue;
}

SEXP loadModel(SEXP r_path) {
  std::string path = std::string(CHAR(STRING_ELT(r_path, 0)));

  ABCBoost::Config* config = new ABCBoost::Config();
  config->from_wrapper = true;
  config->model_pretrained_path = path;
  ABCBoost::ModelHeader model_header = ABCBoost::GradientBoosting::loadModelHeader(config);
  if (model_header.config.null_config == false) {
    *config = model_header.config;
    config->model_mode = "test";
  } else {
    printf("[ERROR] Model file not found: -model (%s)\n",config->model_pretrained_path.c_str());
    return R_NilValue;
  }

  ABCBoost::Data* data = new ABCBoost::Data(config);

  ABCBoost::GradientBoosting* model;
  config->model_use_logit =
      (config->model_name.find("logit") != std::string::npos);

  if (config->model_name == "mart" || config->model_name == "robustlogit") {
    model = new ABCBoost::Mart(data, config);
  } else if (config->model_name == "abcmart" ||
             config->model_name == "abcrobustlogit") {
    model = new ABCBoost::ABCMart(data, config);
  } else if (config->model_name == "regression") {
    config->model_is_regression = true;
    model = new ABCBoost::Regression(data, config);
  } else if (config->model_name == "lambdamart" || config->model_name == "lambdarank") {
    config->model_is_regression = true;
		config->model_use_logit = true;
    model = new ABCBoost::LambdaMart(data, config);
  } else if (config->model_name == "gbrank") {
    config->model_is_regression = 1;
		config->model_use_logit = true;
    model = new ABCBoost::GBRank(data, config);
  } else {
    printf("[ERROR] Unsupported model name %s\n", config->model_name.c_str());
    return R_NilValue;
  }

  config->model_pretrained_path = path;
  config->data_path = path;
  config->load_data_head_only = true;
  config->model_suffix = "";
  
  data->data_header = model_header.auxDataHeader;

  model->init();
  model->loadModel();
  SEXP ret = R_MakeExternalPtr(model, R_NilValue, R_NilValue);
  return ret;
}

SEXP read_libsvm(SEXP r_path) {
  std::string path = std::string(CHAR(STRING_ELT(r_path, 0)));
  
  const char* delimiter = " ,:";
  std::ifstream infile(path);
  std::string line;
  std::vector<std::string> buffer;

  while (getline(infile, line)) {
    buffer.push_back(line);
  }

  int T = 1;
  int n_lines = buffer.size();
  int step = (n_lines + T - 1) / T;

  std::vector<int> is;
  std::vector<int> ps;
  std::vector<double> xs;
  std::vector<double> y;
  int n_feats_local = 0;
  int start = 0, end = n_lines;
  int p_cnt = 0;
  ps.push_back(p_cnt);
  bool one_based_warning = false;
  for (int i = start; i < end; ++i) {
    char* token = &(buffer[i][0]);
    char* ptr;
    char* pos = NULL;
    pos = strtok_r(token, delimiter, &ptr);
    y.push_back(atof(pos));
    int j;
    double j_val;
    bool is_key = true;
    while (true) {
      pos = strtok_r(NULL, delimiter, &ptr);
      if (pos == NULL) break;
      if (is_key) {
        is_key = false;
        j = atoi(pos);
      } else {
        is_key = true;
        j_val = atof(pos);
        
        if(j < 1){
          one_based_warning = true;
          continue;
        }
        is.push_back(j - 1);
        xs.push_back(j_val);
        ++p_cnt;
      }
    }
    ps.push_back(p_cnt);
  }
  if(one_based_warning)
    printf("[Warning] ignored invalid index. Column index must start with the index 1 in libsvm format.\n");
  
  SEXP r_is = PROTECT(allocVector(INTSXP, is.size()));
  for (int i = 0; i < is.size(); ++i) 
    INTEGER(r_is)[i] = is[i];
  SEXP r_ps = PROTECT(allocVector(INTSXP, ps.size()));
  for (int i = 0;i < ps.size(); ++i)
    INTEGER(r_ps)[i] = ps[i];
  SEXP r_xs = PROTECT(allocVector(REALSXP, xs.size()));
  for (int i = 0;i < xs.size(); ++i)
    REAL(r_xs)[i] = xs[i];
  SEXP r_y = PROTECT(allocVector(REALSXP, y.size()));
  for (int i = 0;i < y.size(); ++i)
    REAL(r_y)[i] = y[i];

  const char *names[] = {"is", "ps", "xs", "y", ""};                   /* note the null string */
  SEXP res = PROTECT(mkNamed(VECSXP, names));  /* list of length 4 */
  SET_VECTOR_ELT(res, 0, r_is);
  SET_VECTOR_ELT(res, 1, r_ps);
  SET_VECTOR_ELT(res, 2, r_xs);
  SET_VECTOR_ELT(res, 3, r_y);
  UNPROTECT(5);
  return res;
}

}

