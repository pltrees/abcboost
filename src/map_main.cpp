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
  ABCBoost::Config config = ABCBoost::Config();
  config.parseArguments(argc, argv);
  config.model_mode = "train";

  ABCBoost::Data data = ABCBoost::Data(&config);
  data.loadData();
  int data_name_start_nix = config.data_path.find_last_of('/') + 1;
  int data_name_start_win = config.data_path.find_last_of('\\') + 1;
  int data_name_start = std::max(data_name_start_nix, data_name_start_win);
  std::string data_name = config.data_path.substr(data_name_start);
  auto data_out =
      fopen((config.experiment_folder + data_name + ".map").c_str(), "wb");
  data.saveData(data_out);
  fclose(data_out);
  printf("Mapping is created at: %s\n",
         (config.experiment_folder + data_name + ".map").c_str());
  if(config.map_dump_format != ""){
    if(config.map_dump_format != "libsvm" && config.map_dump_format != "csv"){
      printf("[ERROR] Unsuported dump format (%s), which must be libsvm or csv\n",config.map_dump_format.c_str());
    }else{
      auto dump_out = fopen((config.experiment_folder + data_name + "." + config.map_dump_format).c_str(), "w");
      data.dumpData(dump_out,config.map_dump_format);
      fclose(dump_out);
      printf("Discretized data are dumped at: %s\n",
            (config.experiment_folder + data_name + "." + config.map_dump_format).c_str());
    }
  }
  return 0;
}
