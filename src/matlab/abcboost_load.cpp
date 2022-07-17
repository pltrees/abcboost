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

	char* s = mxArrayToString(prhs[0]);
	std::string path = std::string(s);
  plhs[0] = mxCreateNumericMatrix(1,1,mxINT64_CLASS,mxREAL);

	ABCBoost::Config* config = new ABCBoost::Config();
	config->model_pretrained_path = path;
	ABCBoost::ModelHeader model_header = ABCBoost::GradientBoosting::loadModelHeader(config);
	if(model_header.config.null_config == false){
		*config = model_header.config;
		config->model_mode = "test";
	}else{
    printf("[ERROR] Model file not found: -model (%s)\n",config->model_pretrained_path.c_str());
    return;
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

	config->model_pretrained_path = path;
	config->data_path = path;
	config->from_wrapper = true;
	config->load_data_head_only = true;
	config->model_suffix = "";

	data->data_header = model_header.auxDataHeader;

	model->init();
	model->loadModel();
	uint64_t* ip = (uint64_t*) mxGetData(plhs[0]);
	*ip = reinterpret_cast<uint64_t>(model);
}

