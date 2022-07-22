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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
namespace py = pybind11;

extern "C"{
void* train(py::array_t<double> Y, py::object general_X, std::string model_name, int iter, int leaves, double shrinkage, py::kwargs params = py::none()) {
  std::string class_name = py::str(general_X.get_type());
  py::array_t<double> X;
  std::vector<std::vector<std::pair<int,double>>> kvs;
  bool is_sparse = false;
  py::module scipy;
  try{
    scipy = py::module::import("scipy.sparse");
    if(scipy.attr("isspmatrix")(general_X).cast<bool>() == true){
      is_sparse = true;
    }
  }catch(...){

  }

  int n_row = 0;
  int n_col = 0;
  if(class_name == "<class 'numpy.ndarray'>"){
    int dim = py::array_t<double>(general_X).ndim();
    if(dim != 2){
      py::print("X should be 2-dimensional. Found",dim);
      return nullptr;
    }
    n_row = py::array_t<double>(general_X).shape(0);
    n_col = py::array_t<double>(general_X).shape(1);
    py::object sh = general_X.attr("shape");
    X = general_X.attr("ravel")(py::str("F")).attr("reshape")(sh);
  }else if (is_sparse){
    py::object csr = general_X.attr("tocsr")();
    py::array_t<int> idx = py::array_t<int>(csr.attr("indices"));
    int* pidx = (int*)idx.request().ptr;
    py::array_t<int> indptr = py::array_t<int>(csr.attr("indptr"));
    int* pindptr = (int*)indptr.request().ptr;
    py::array_t<double> data = py::array_t<double>(csr.attr("data"));
    double* pdata = (double*)data.request().ptr;
    for(int i = 0;i < indptr.size() - 1;++i){
      std::vector<std::pair<int,double>> vec;
      for(int j = pindptr[i];j < pindptr[i + 1];++j){
        vec.push_back(std::make_pair(pidx[j] + 1,pdata[j]));
      }
      kvs.push_back(vec);
    }
    py::object sh = csr.attr("shape");
    n_row = sh.attr("__getitem__")(0).cast<int>();
    n_col = sh.attr("__getitem__")(1).cast<int>();
  }else{
    py::print("Unrecognized matrix X:", py::str(general_X.get_type()));
    return nullptr;
  }


  ABCBoost::Config* config = new ABCBoost::Config();
  config->model_mode = "train";
  config->from_wrapper = true;
  config->model_mapping_name = "tmp" + std::to_string((unsigned int)clock()) + config->mapping_suffix;
  config->model_name = model_name;
  config->model_n_iterations = iter;
  config->tree_max_n_leaves = leaves;
  config->model_shrinkage = shrinkage;

  config->model_pretrained_path = "";
  config->save_model = false;
  config->save_log = false;

  if (params != py::none()){
    for(auto it : params){
      const std::string& name = py::cast<const std::string>(it.first);
      std::string val = py::cast<const std::string>(py::str(it.second));
      try{
        config->parse(name,val);
      }catch (...){
        py::print("The program stopped due to the above error(s).");
        return nullptr;
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
    return nullptr;
  }

  py::array_t<double> Ycopy = Y.attr("copy")();
  config->mem_Y_matrix = static_cast<double *>(Ycopy.request().ptr);
  if(is_sparse){
    config->mem_X_kv = kvs;
    config->mem_is_sparse = true;
  }else{
    config->mem_X_matrix = static_cast<double *>(X.request().ptr);
    config->mem_is_sparse = false;
  }
  config->mem_n_row = n_row;
  config->mem_n_col = n_col;

  data->loadData(true);
  model->init();
  model->loadModel();
  model->setupExperiment();

  model->train();
  return model;
}

py::dict test(py::array_t<double> Y, py::object general_X, void* py_model, py::kwargs params = py::none()) {
  std::string class_name = py::str(general_X.get_type());
  py::array_t<double> X;
  std::vector<std::vector<std::pair<int,double>>> kvs;
  bool is_sparse = false;
  py::module scipy;
  try{
    scipy = py::module::import("scipy.sparse");
    if(scipy.attr("isspmatrix")(general_X).cast<bool>() == true){
      is_sparse = true;
    }
  }catch(...){

  }

  int n_row = 0;
  int n_col = 0;
  if(class_name == "<class 'numpy.ndarray'>"){
    int dim = py::array_t<double>(general_X).ndim();
    if(dim != 2){
      py::print("X should be 2-dimensional. Found",dim);
      return py::dict();
    }
    n_row = py::array_t<double>(general_X).shape(0);
    n_col = py::array_t<double>(general_X).shape(1);
    py::object sh = general_X.attr("shape");
    X = general_X.attr("ravel")(py::str("F")).attr("reshape")(sh);
  }else if (is_sparse){
    py::object csr = general_X.attr("tocsr")();
    py::array_t<int> idx = py::array_t<int>(csr.attr("indices"));
    int* pidx = (int*)idx.request().ptr;
    py::array_t<int> indptr = py::array_t<int>(csr.attr("indptr"));
    int* pindptr = (int*)indptr.request().ptr;
    py::array_t<double> data = py::array_t<double>(csr.attr("data"));
    double* pdata = (double*)data.request().ptr;
    for(int i = 0;i < indptr.size() - 1;++i){
      std::vector<std::pair<int,double>> vec;
      for(int j = pindptr[i];j < pindptr[i + 1];++j){
        vec.push_back(std::make_pair(pidx[j] + 1,pdata[j]));
      }
      kvs.push_back(vec);
    }
    py::object sh = csr.attr("shape");
    n_row = sh.attr("__getitem__")(0).cast<int>();
    n_col = sh.attr("__getitem__")(1).cast<int>();
  }else{
    py::print("Unrecognized matrix X:", py::str(general_X.get_type()));
    return py::dict();
  }

  ABCBoost::GradientBoosting* model =
      reinterpret_cast<ABCBoost::GradientBoosting*>(py_model);
  ABCBoost::Config* config = model->getConfig();
  if (params != py::none()){
    for(auto it : params){
      const std::string& name = py::cast<const std::string>(it.first);
      std::string val = py::cast<const std::string>(py::str(it.second));
      try{
        config->parse(name,val);
      }catch (...){
        py::print("The program stopped due to the above error(s).");
        return py::dict();
      }
    }
  }
  config->model_mode = "test";

  py::array_t<double> Ycopy = Y.attr("copy")();
  config->mem_Y_matrix = (double*)Ycopy.request().ptr;
  if(is_sparse){
    config->mem_X_kv = kvs;
    config->mem_is_sparse = true;
  }else{
    config->mem_X_matrix = static_cast<double *>(X.request().ptr);
    config->mem_is_sparse = false;
  }
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

  py::dict ret;
  py::array_t<double> py_pred = py::array_t<double>(n_row);
  double* buff = (double*)py_pred.request().ptr;
  for (int i = 0; i < n_row; ++i){
    buff[i] = prediction[i];
  }
  ret["prediction"] = py_pred;
  
  int testlog_col = model->testlog.size() > 0 ? model->testlog[0].size() : 0;
  int testlog_row = model->testlog.size();
  py::array_t<double> py_testlog = py::array_t<double>(testlog_row * testlog_col);
  buff = (double*)py_testlog.request().ptr;
  for(int i = 0;i < testlog_row;++i){
    for(int j = 0;j < model->testlog[i].size();++j){
      buff[i * testlog_col + j] = model->testlog[i][j];
    }
  }
  py_testlog.resize({testlog_row,testlog_col});
  ret["testlog"] = py_testlog;

  if(config->save_prob){
    py::array_t<double> py_prob = py::array_t<double>(n_row * n_classes);
    double* buff = (double*)py_prob.request().ptr;
    for (int j = 0; j < n_classes; ++j){
      for (int i = 0; i < n_row; ++i){
        buff[i * n_classes + j] = prob[j * n_row + i];
      }
    }
    py_prob.resize({n_row,n_classes});
    ret["probability"] = py_prob;
  }
  model->testlog.clear();
  return ret;
}

py::dict predict(py::object general_X, void* py_model, py::kwargs params = py::none()) {
  std::string class_name = py::str(general_X.get_type());
  py::array_t<double> X;
  std::vector<std::vector<std::pair<int,double>>> kvs;
  bool is_sparse = false;
  py::module scipy;
  try{
    scipy = py::module::import("scipy.sparse");
    if(scipy.attr("isspmatrix")(general_X).cast<bool>() == true){
      is_sparse = true;
    }
  }catch(...){

  }

  int n_row = 0;
  int n_col = 0;
  if(class_name == "<class 'numpy.ndarray'>"){
    int dim = py::array_t<double>(general_X).ndim();
    if(dim != 2){
      py::print("X should be 2-dimensional. Found",dim);
      return py::dict();
    }
    n_row = py::array_t<double>(general_X).shape(0);
    n_col = py::array_t<double>(general_X).shape(1);
    py::object sh = general_X.attr("shape");
    X = general_X.attr("ravel")(py::str("F")).attr("reshape")(sh);
  }else if (is_sparse){
    py::object csr = general_X.attr("tocsr")();
    py::array_t<int> idx = py::array_t<int>(csr.attr("indices"));
    int* pidx = (int*)idx.request().ptr;
    py::array_t<int> indptr = py::array_t<int>(csr.attr("indptr"));
    int* pindptr = (int*)indptr.request().ptr;
    py::array_t<double> data = py::array_t<double>(csr.attr("data"));
    double* pdata = (double*)data.request().ptr;
    for(int i = 0;i < indptr.size() - 1;++i){
      std::vector<std::pair<int,double>> vec;
      for(int j = pindptr[i];j < pindptr[i + 1];++j){
        vec.push_back(std::make_pair(pidx[j] + 1,pdata[j]));
      }
      kvs.push_back(vec);
    }
    py::object sh = csr.attr("shape");
    n_row = sh.attr("__getitem__")(0).cast<int>();
    n_col = sh.attr("__getitem__")(1).cast<int>();
  }else{
    py::print("Unrecognized matrix X:", py::str(general_X.get_type()));
    return py::dict();
  }

  ABCBoost::GradientBoosting* model =
      reinterpret_cast<ABCBoost::GradientBoosting*>(py_model);
  ABCBoost::Config* config = model->getConfig();

  if (params != py::none()){
    for(auto it : params){
      const std::string& name = py::cast<const std::string>(it.first);
      std::string val = py::cast<const std::string>(py::str(it.second));
      try{
        config->parse(name,val);
      }catch (...){
        py::print("The program stopped due to the above error(s).");
        return py::dict();
      }
    }
  }

  config->model_mode = "test";

  config->mem_Y_matrix = NULL;
  if(is_sparse){
    config->mem_X_kv = kvs;
    config->mem_is_sparse = true;
  }else{
    config->mem_X_matrix = static_cast<double *>(X.request().ptr);
    config->mem_is_sparse = false;
  }
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

  py::dict ret;
  py::array_t<double> py_pred = py::array_t<double>(n_row);
  double* buff = (double*)py_pred.request().ptr;
  for (int i = 0; i < n_row; ++i){
    buff[i] = prediction[i];
  }
  ret["prediction"] = py_pred;

  if(config->save_prob){
    py::array_t<double> py_prob = py::array_t<double>(n_row * n_classes);
    double* buff = (double*)py_prob.request().ptr;
    for (int j = 0; j < n_classes; ++j){
      for (int i = 0; i < n_row; ++i){
        buff[i * n_classes + j] = prob[j * n_row + i];
      }
    }
    py_prob.resize({n_row,n_classes});
    ret["probability"] = py_prob;
  }

  config->no_label = prev_no_label;
  return ret;
}

void saveModel(void* py_model, std::string path) {
  ABCBoost::GradientBoosting* model =
      reinterpret_cast<ABCBoost::GradientBoosting*>(py_model);
  model->setExperimentPath(path);
  model->getConfig()->from_wrapper = true;
  model->getConfig()->model_suffix = "";

  model->saveModel(model->getConfig()->model_n_iterations);
}

void* loadModel(std::string path) {
  ABCBoost::Config* config = new ABCBoost::Config();
  config->from_wrapper = true;
  config->model_pretrained_path = path;
  ABCBoost::ModelHeader model_header = ABCBoost::GradientBoosting::loadModelHeader(config);
  if (model_header.config.null_config == false) {
    *config = model_header.config;
    config->model_mode = "test";
  } else {
    printf("[ERROR] Model file not found: -model (%s)\n",config->model_pretrained_path.c_str());
    return nullptr;
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
    return nullptr;
  }

  config->model_pretrained_path = path;
  config->data_path = path;
  config->load_data_head_only = true;
  config->model_suffix = "";
  
  data->data_header = model_header.auxDataHeader;

  model->init();
  model->loadModel();
  return model;
}
}

PYBIND11_MODULE(abcboost, m) {
    m.doc() = "ABCBoost is a high-performance gradient boosting library for adaptive base class training."; // optional module docstring
    m.def("train", &train, "train ABCBoost model",
          py::arg("Y"),
          py::arg("X"),
          py::arg("model_name"),
          py::arg("iter"),
          py::arg("leaves"),
          py::arg("shrinkage")
          );
    m.def("test",&test, "test ABCBoost model",
          py::arg("Y"),
          py::arg("X"),
          py::arg("model"));
    m.def("predict",&predict, "predict ABCBoost model",
          py::arg("X"),
          py::arg("model"));
    m.def("load",&loadModel,"load ABCBoost model",
          py::arg("path"));
    m.def("save",&saveModel,"save ABCBoost model",
          py::arg("model"),
          py::arg("path"));
}

