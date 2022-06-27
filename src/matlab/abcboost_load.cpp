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
		printf("config->model_mapping_name (%s)\n",config->model_mapping_name.c_str());
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
  } else {
    printf("Unsupported model name %s\n", config->model_name.c_str());
  }

	config->model_pretrained_path = path;
	config->data_path = path;
	config->from_wrapper = true;
	config->load_data_head_only = true;
	config->model_suffix = "";

	std::string mapping_name = config->model_mapping_name;
	FILE* fp = fopen(mapping_name.c_str(),"rb");
	if(fp == NULL){
		printf("[ERROR] mapping file not found! (%s)\n",mapping_name.c_str());
		uint64_t* ip = (uint64_t*) mxGetData(plhs[0]);
		*ip = reinterpret_cast<uint64_t>(nullptr);
		return;
	}
	data->data_header = ABCBoost::DataHeader::deserialize(fp);
	fclose(fp);

	model->init();
	model->loadModel();
	uint64_t* ip = (uint64_t*) mxGetData(plhs[0]);
	*ip = reinterpret_cast<uint64_t>(model);
}

