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
  int X_n_row = mxGetM(prhs[0]);
  int X_n_col = mxGetN(prhs[0]);
	uint64_t* pmodel= (uint64_t*)mxGetData(prhs[1]);
	ABCBoost::GradientBoosting* model = reinterpret_cast<ABCBoost::GradientBoosting*>(*pmodel);

  /* Create matrix for the return argument. */
  plhs[0] = mxCreateNumericMatrix(1,1,mxINT64_CLASS,mxREAL);
  
  /* Assign pointers to each input and output. */
  
	ABCBoost::Config* config = model->getConfig();
  if (nrhs > 2){
    int nfields = mxGetNumberOfFields(prhs[2]);
    int nelems = mxGetNumberOfElements(prhs[2]);
    if(nelems != 1){
      printf("[ERROR] The param structure parameter must have one element. The #elements we got is %d\n",nelems);
      return;
    }
    for(int i = 0;i < nfields;++i){
      mxArray* field = mxGetFieldByNumber(prhs[2], 0, i);
      std::string name = std::string(mxGetFieldNameByNumber(prhs[2],i));
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


	config->mem_Y_matrix = NULL;

  if(mxIsSparse(prhs[0])){
    config->mem_X_kv = organize_sparse(prhs[0]);
    config->mem_is_sparse = true;
  }else{
    double* X = mxGetPr(prhs[0]);
    config->mem_X_matrix = X;
    config->mem_is_sparse = false;
  }
	config->mem_n_row = X_n_row;
	config->mem_n_col = X_n_col;
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
  std::vector<double> prediction(X_n_row);
  if(config->save_prob){
    prob.resize(X_n_row * n_classes);
  }
	model->returnPrediction(prediction.data(),prob.data());


  const char* cc_field_names[] = {"prediction","probability"};
  int len = 1;
  if(config->save_prob){
    len = 2;
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

  if(config->save_prob){
    auto* ret_prob = mxCreateDoubleMatrix(X_n_row, n_classes, mxREAL);
    double* p_ret_prob = mxGetPr(ret_prob);
    for(int i = 0;i < X_n_row * n_classes;++i)
      p_ret_prob[i] = prob[i];
    mxSetFieldByNumber(plhs[0],0,1,ret_prob);
  }
  config->no_label = prev_no_label;
}

