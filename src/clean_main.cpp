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
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "data.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  std::unique_ptr<ABCBoost::Config> config =
      std::unique_ptr<ABCBoost::Config>(new ABCBoost::Config());
  config->parseArguments(argc, argv);
  config->model_mode = "clean";
  config->sanityCheck();


  std::unique_ptr<ABCBoost::Data> data =
      std::unique_ptr<ABCBoost::Data>(new ABCBoost::Data(config.get()));
  if(config->clean_info == ""){
    data->cleanCSV();
  }else{
    if (!data->doesFileExist(config->clean_info)) {
      fprintf(stderr, "[ERROR] Specified cleaninfo file (%s) does not exist!\n",config->clean_info.c_str());
      exit(1);
    }
    FILE* fp = fopen(config->clean_info.c_str(),"rb");
    data->deserializeCleanInfo(fp);
    fclose(fp);
    data->cleanCSVwithInfo();
  }
  return 0;
}

