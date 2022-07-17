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

int main(int argc, char* argv[]) {
  std::unique_ptr<ABCBoost::Config> config =
      std::unique_ptr<ABCBoost::Config>(new ABCBoost::Config());
  config->parseArguments(argc, argv);
  config->model_mode = "test";

  ABCBoost::ModelHeader model_header =
      ABCBoost::GradientBoosting::loadModelHeader(config.get());
  if (model_header.config.null_config == false) {
    *config = model_header.config;
    config->parseArguments(argc, argv);
    config->model_mode = "test";
  } else {
    printf("[ERROR] Model file not found: -model (%s)\n",config->model_pretrained_path.c_str());
    exit(1);
  }

  config->sanityCheck();

  std::unique_ptr<ABCBoost::Data> data =
      std::unique_ptr<ABCBoost::Data>(new ABCBoost::Data(config.get()));
  data->data_header = model_header.auxDataHeader;
  data->loadData(false);

  std::unique_ptr<ABCBoost::GradientBoosting> model;

  config->model_use_logit =
      (config->model_name.find("logit") != std::string::npos);

  if (config->model_name == "mart" || config->model_name == "robustlogit") {
    if(data->data_header.n_classes == 2){
      model = std::unique_ptr<ABCBoost::GradientBoosting>(
          new ABCBoost::BinaryMart(data.get(), config.get()));
    }else{
      model = std::unique_ptr<ABCBoost::GradientBoosting>(
          new ABCBoost::Mart(data.get(), config.get()));
    }
  } else if (config->model_name == "abcmart" ||
             config->model_name == "abcrobustlogit") {
    if(data->data_header.n_classes == 2){
      model = std::unique_ptr<ABCBoost::GradientBoosting>(
          new ABCBoost::BinaryMart(data.get(), config.get()));
    }else{
      model = std::unique_ptr<ABCBoost::GradientBoosting>(
          new ABCBoost::ABCMart(data.get(), config.get()));
    }
  } else if (config->model_name == "regression") {
    config->model_is_regression = true;
    model = std::unique_ptr<ABCBoost::GradientBoosting>(
        new ABCBoost::Regression(data.get(), config.get()));
  } else if (config->model_name == "lambdamart" || config->model_name == "lambdarank") {
    config->model_is_regression = 1;
		config->model_use_logit = true;
    model = std::unique_ptr<ABCBoost::GradientBoosting>(
				new ABCBoost::LambdaMart(data.get(), config.get()));
  } else if (config->model_name == "gbrank") {
    config->model_is_regression = 1;
		config->model_use_logit = true;
    model = std::unique_ptr<ABCBoost::GradientBoosting>(
				new ABCBoost::GBRank(data.get(), config.get()));
  } else {
    fprintf(stderr, "Unsupported model name %s\n", config->model_name.c_str());
    exit(1);
  }

  model->init();
  model->loadModel();
  model->setupExperiment();

  model->test();
  model->savePrediction();
  return 0;
}
