#include<stdio.h>
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
#include "mex.h"
#include "matlab_utils.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[]){
  /* The input must be a noncomplex scalar double.*/
  int Y_n_row = mxGetM(prhs[0]);
  int X_n_row = mxGetM(prhs[1]);
  int X_n_col = mxGetN(prhs[1]);
  char* s = mxArrayToString(prhs[2]);
  std::string model_name = std::string(s);

  /* Create matrix for the return argument. */
  plhs[0] = mxCreateNumericMatrix(1,1,mxINT64_CLASS,mxREAL);
  
  /* Assign pointers to each input and output. */
  
  ABCBoost::Config* config = new ABCBoost::Config();
  config->model_mode = "train";
  config->from_wrapper = true;
  config->save_model = false;
  config->save_log = false;
  config->model_mapping_name = "tmp" + std::to_string((unsigned int)clock()) + config->mapping_suffix;
  config->model_name = model_name;
  if (nrhs > 3)
    config->model_n_iterations = mxGetScalar(prhs[3]);
  if (nrhs > 4)
    config->tree_max_n_leaves = mxGetScalar(prhs[4]);
  if (nrhs > 5)
    config->model_shrinkage = mxGetScalar(prhs[5]);

  if (nrhs > 6){
    int nfields = mxGetNumberOfFields(prhs[6]);
    int nelems = mxGetNumberOfElements(prhs[6]);
    if(nelems != 1){
      printf("[ERROR] The param structure parameter must have one element. The #elements we got is %d\n",nelems);
      return;
    }
    for(int i = 0;i < nfields;++i){
      mxArray* field = mxGetFieldByNumber(prhs[6], 0, i);
      std::string name = std::string(mxGetFieldNameByNumber(prhs[6],i));
      std::string val;
      if(mxIsNumeric(field)){
        val = std::to_string(mxGetScalar(field));
      }else if(mxIsChar(field)){
        val = std::string(mxArrayToString(field));
      }else{
        printf("[Error] The param has to be either scalar or string");
        return;
      }
      try{
        config->parse(name,val);
      }catch (...){
        mexErrMsgTxt("The program stopped due to the above error(s).");
        return;
      }
    }
  }

  ABCBoost::Data* data = new ABCBoost::Data(config);

  ABCBoost::GradientBoosting *model;
  config->model_use_logit = (config->model_name.find("logit") != std::string::npos);

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
    return;
  }

  config->model_pretrained_path = "";
  
  double* Y = mxGetPr(prhs[0]);
  config->mem_Y_matrix = Y;
  
  if(mxIsSparse(prhs[1])){
    config->mem_X_kv = organize_sparse(prhs[1]);
    config->mem_is_sparse = true;
  }else{
    double* X = mxGetPr(prhs[1]);
    config->mem_X_matrix = X;
    config->mem_is_sparse = false;
  }
  config->mem_n_row = X_n_row;
  config->mem_n_col = X_n_col;
  
  data->loadData(true);
  model->init();
  model->loadModel();
  model->setupExperiment();
  
  model->train();
  uint64_t* ip = (uint64_t*) mxGetData(plhs[0]);
  *ip = reinterpret_cast<uint64_t>(model);
}

