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
	std::vector<double> ans(X_n_row * n_classes);
	model->returnPrediction(ans.data());
	plhs[0] = mxCreateDoubleMatrix(X_n_row, n_classes, mxREAL);
	double* ret = mxGetPr(plhs[0]);
	for(int i = 0;i < X_n_row * n_classes;++i)
		ret[i] = ans[i];
  config->no_label = prev_no_label;
}

