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
	uint64_t* pmodel= (uint64_t*)mxGetData(prhs[2]);
	ABCBoost::GradientBoosting* model = reinterpret_cast<ABCBoost::GradientBoosting*>(*pmodel);

  /* Create matrix for the return argument. */
  plhs[0] = mxCreateNumericMatrix(1,1,mxINT64_CLASS,mxREAL);
  
  /* Assign pointers to each input and output. */
  
	ABCBoost::Config* config = model->getConfig();

  if (nrhs > 3){
    int nfields = mxGetNumberOfFields(prhs[3]);
    int nelems = mxGetNumberOfElements(prhs[3]);
    if(nelems != 1){
      printf("[ERROR] The param structure parameter must have one element. The #elements we got is %d\n",nelems);
      return;
    }
    for(int i = 0;i < nfields;++i){
      mxArray* field = mxGetFieldByNumber(prhs[3], 0, i);
      std::string name = std::string(mxGetFieldNameByNumber(prhs[3],i));
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

	config->model_mode = "test";

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
	config->from_wrapper = true;
  config->save_log = false;

	model->getData()->loadData(true);
	model->init();
	model->setupExperiment();

  model->testlog.clear();

	model->test();
	int n_classes = model->getData()->data_header.n_classes;
	std::vector<double> prob(1);
  std::vector<double> prediction(X_n_row);
  if(config->save_prob){
    prob.resize(X_n_row * n_classes);
  }
	model->returnPrediction(prediction.data(),prob.data());

  const char* cc_field_names[] = {"prediction","testlog","probability"};
  int len = 2;
  if(config->save_prob){
    len = 3;
  }
  char** field_names = new char*[len];
  for(int i = 0;i < len;++i){
    field_names[i] = (char*)mxMalloc(20);
    memcpy(field_names[i],cc_field_names[i],(strlen(cc_field_names[i]) + 1) * sizeof(char));
  }

  plhs[0] = mxCreateStructMatrix(1,1,len,(const char**)field_names);
  for(int i = 0;i < len;++i)
    mxFree(field_names[i]);
  delete[] field_names;

  auto* ret_prediction = mxCreateDoubleMatrix(X_n_row,1,mxREAL);
  double* p_ret_prediction = mxGetPr(ret_prediction);
  for(int i = 0;i < X_n_row;++i)
    p_ret_prediction[i] = prediction[i];
  mxSetFieldByNumber(plhs[0],0,0,ret_prediction);

  int testlog_col = model->testlog.size() > 0 ? model->testlog[0].size() : 0;
  int testlog_row = model->testlog.size();
  auto* ret_testlog = mxCreateDoubleMatrix(testlog_row,testlog_col,mxREAL);
  double* p_ret_testlog = mxGetPr(ret_testlog);
  for(int i = 0;i < testlog_row;++i){
    for(int j = 0;j < model->testlog[i].size();++j){
      p_ret_testlog[j * testlog_row + i] = model->testlog[i][j];
    }
  }
  mxSetFieldByNumber(plhs[0],0,1,ret_testlog);

  if(config->save_prob){
    auto* ret_prob = mxCreateDoubleMatrix(X_n_row, n_classes, mxREAL);
    double* p_ret_prob = mxGetPr(ret_prob);
    for(int i = 0;i < X_n_row * n_classes;++i)
      p_ret_prob[i] = prob[i];
    mxSetFieldByNumber(plhs[0],0,2,ret_prob);
  }
  model->testlog.clear();
}

