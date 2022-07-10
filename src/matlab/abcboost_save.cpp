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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[]){
  /* The input must be a noncomplex scalar double.*/
	uint64_t* pmodel= (uint64_t*)mxGetData(prhs[0]);
	ABCBoost::GradientBoosting* model = reinterpret_cast<ABCBoost::GradientBoosting*>(*pmodel);

	char* s = mxArrayToString(prhs[1]);
	std::string path = std::string(s);
	model->setExperimentPath(path);
	model->getConfig()->from_wrapper = true;
	model->getConfig()->model_suffix = "";

	model->saveModel(model->getConfig()->model_n_iterations);
}

