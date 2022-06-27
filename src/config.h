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

#ifndef ABCBOOST_CONFIG_H
#define ABCBOOST_CONFIG_H

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#ifdef USE_MEX_CMD
#include "mex.h"
#define printf mexPrintf
#endif

#ifdef USE_R_CMD
#include <R.h>
#define printf Rprintf
#endif

namespace ABCBoost {
class Config {
 public:
  // Data config
  bool data_is_matrix = false;
  bool data_use_mean_as_missing = false;
  double data_min_bin_size = 1e-10;  // minimum size of the bin
  double data_sparsity_threshold = 0.33;
  int data_max_n_bins = 1000;  // max number of bins
  int data_n_feats = 0;
  std::string data_path = "";
  bool from_wrapper = false;
  bool load_data_head_only = false;
  double* mem_Y_matrix = NULL;
  double* mem_X_matrix = NULL;
  int mem_n_row = 0;
  int mem_n_col = 0;
  std::vector<std::vector<std::pair<int,double>>> mem_X_kv;
  bool mem_is_sparse = false;
  std::string model_suffix = ".model";
  std::string mapping_suffix = ".map";

  // Tree config
  bool tree_fast_bin_sort = false;
  double tree_clip_value = 100;
  double tree_damping_factor = 1e-300;
  int tree_max_n_leaves = 4;
  int tree_min_node_size = 10;

  // Model config
  bool model_is_regression = false;
  bool model_use_logit = true;
  bool model_use_weighted_update = true;
  double model_data_sample_rate = 1.0;
  double model_feature_sample_rate = 1.0;
  double model_shrinkage = 0.08;
  bool model_n_iter_cmd = false;
  int model_n_iterations = 1000;
  int model_more_iter = 0;
  int model_save_every = 500;
  int model_eval_every = 1;
  std::string model_mode = "train";
  std::string model_name = "abcrobustlogit";
  std::string model_pretrained_path = "";
  std::string model_gap = "0";
  std::string model_mapping_name = "";
  bool null_config = false;

  double abc_sample_rate = 1;
  int abc_sample_min_data = 2000;
  int warmup_iter = 0;
  bool warmup_use_logit = true;

  bool regression_l1_loss = false;
  bool regression_huber_loss = false;
  bool regression_use_hessian = true;
  bool regression_auto_clip_value = true;
  double huber_delta = 1.0;
  double regression_lp_loss = 2.0;

  // Parallelism config
  int use_gpu = 1;
  int use_omp = 0;
  int n_threads = 1;

  // Others
  bool save_log = true;
  bool save_model = true;
  std::string experiment_folder = "./";
  std::string additional_files = "";
  bool no_map = false;
  bool no_label = false;
  double stop_tolerance = 1e-13;
  double regression_stop_factor = 1e-5;


  // Rank Query File
  std::string rank_query_file = "";
  std::string prediction_file = "";

  int base_candidates_size = 1;

  Config(const char* path = "config.txt") {
    std::ifstream file(path);
    if (file.is_open()) {
      std::string line;
      while (getline(file, line)) {
        line.erase(remove_if(line.begin(), line.end(), isspace), line.end());
        if (line[0] == '#' || line.empty()) continue;
        auto delimiter_pos = line.find("=");
        auto key = line.substr(0, delimiter_pos);
        auto value = line.substr(delimiter_pos + 1);
        parse(key, value);
      }
      file.close();
    } else {
      // config.txt not found: using default parameters values as in config.h
      // std::cout << "Unable to open config file." << '\n';
    }
  }

  void saveString(std::string str, FILE* fp) {
    size_t size = str.length();
    fwrite(&size, sizeof(size), 1, fp);
    fwrite(str.c_str(), size, 1, fp);
  }

  void loadString(std::string& str, FILE* fp) {
    size_t size;
    auto ret = 0;
    ret = fread(&size, sizeof(size), 1, fp);
    str.resize(size);
    ret += fread(&str[0], size, 1, fp);
  }

  void serialize(FILE* fp) {
    fwrite(&data_is_matrix, sizeof(bool), 1, fp);
    fwrite(&data_use_mean_as_missing, sizeof(bool), 1, fp);
    fwrite(&data_min_bin_size, sizeof(double), 1,
           fp);  // minimum size of the bin
    fwrite(&data_sparsity_threshold, sizeof(double), 1, fp);
    fwrite(&data_max_n_bins, sizeof(int), 1, fp);  // max number of bins
    fwrite(&data_n_feats, sizeof(int), 1, fp);
    saveString(data_path, fp);

    // Tree config
    fwrite(&tree_fast_bin_sort, sizeof(bool), 1, fp);
    fwrite(&tree_clip_value, sizeof(int), 1, fp);
    fwrite(&tree_damping_factor, sizeof(double), 1, fp);
    fwrite(&tree_max_n_leaves, sizeof(int), 1, fp);
    fwrite(&tree_min_node_size, sizeof(int), 1, fp);

    // Model config
    fwrite(&model_is_regression, sizeof(bool), 1, fp);
    fwrite(&model_use_logit, sizeof(bool), 1, fp);
    fwrite(&model_use_weighted_update, sizeof(bool), 1, fp);
    fwrite(&model_data_sample_rate, sizeof(double), 1, fp);
    fwrite(&model_feature_sample_rate, sizeof(double), 1, fp);
    fwrite(&model_shrinkage, sizeof(double), 1, fp);
    fwrite(&model_n_iterations, sizeof(int), 1, fp);
    fwrite(&model_save_every, sizeof(int), 1, fp);
    fwrite(&model_eval_every, sizeof(int), 1, fp);
    saveString(model_mode, fp);
    saveString(model_name, fp);
    saveString(model_pretrained_path, fp);
    saveString(model_gap, fp);
    saveString(model_mapping_name, fp);
    
    fwrite(&abc_sample_rate, sizeof(double), 1, fp);
    fwrite(&abc_sample_min_data, sizeof(int), 1, fp);
    fwrite(&warmup_iter, sizeof(int), 1, fp);
    fwrite(&warmup_use_logit, sizeof(bool), 1, fp);
    
    fwrite(&regression_l1_loss, sizeof(bool), 1, fp);
    fwrite(&regression_huber_loss, sizeof(bool), 1, fp);
    fwrite(&regression_use_hessian, sizeof(bool), 1, fp);
    fwrite(&regression_auto_clip_value, sizeof(bool), 1, fp);
    fwrite(&huber_delta, sizeof(double), 1, fp);
    fwrite(&regression_lp_loss, sizeof(double), 1, fp);

    // Parallelism config
    fwrite(&use_gpu, sizeof(int), 1, fp);
    fwrite(&use_omp, sizeof(int), 1, fp);
    fwrite(&n_threads, sizeof(int), 1, fp);

    // Others
    fwrite(&save_log, sizeof(bool), 1, fp);
    fwrite(&save_model, sizeof(bool), 1, fp);
    saveString(experiment_folder, fp);
    saveString(additional_files, fp);
    fwrite(&no_map, sizeof(bool), 1, fp);
    fwrite(&stop_tolerance, sizeof(double), 1, fp);
    fwrite(&regression_stop_factor, sizeof(double), 1, fp);

    saveString(rank_query_file, fp);
    saveString(prediction_file, fp);

    // ABC related
    fwrite(&base_candidates_size, sizeof(int), 1, fp);
    
  }

  static Config deserialize(FILE* fp) {
    Config config;
    config.loadConfig(fp);
    return config;
  }

  void loadConfig(FILE* fp) {
    auto ret = 0;
    ret += fread(&data_is_matrix, sizeof(bool), 1, fp);
    ret += fread(&data_use_mean_as_missing, sizeof(bool), 1, fp);
    ret += fread(&data_min_bin_size, sizeof(double), 1,
                 fp);  // minimum size of the bin
    ret += fread(&data_sparsity_threshold, sizeof(double), 1, fp);
    ret += fread(&data_max_n_bins, sizeof(int), 1, fp);  // max number of bins
    ret += fread(&data_n_feats, sizeof(int), 1, fp);
    std::string str;
    loadString(str, fp);

    // Tree config
    ret += fread(&tree_fast_bin_sort, sizeof(bool), 1, fp);
    ret += fread(&tree_clip_value, sizeof(int), 1, fp);
    ret += fread(&tree_damping_factor, sizeof(double), 1, fp);
    ret += fread(&tree_max_n_leaves, sizeof(int), 1, fp);
    ret += fread(&tree_min_node_size, sizeof(int), 1, fp);

    // Model config
    ret += fread(&model_is_regression, sizeof(bool), 1, fp);
    ret += fread(&model_use_logit, sizeof(bool), 1, fp);
    ret += fread(&model_use_weighted_update, sizeof(bool), 1, fp);
    ret += fread(&model_data_sample_rate, sizeof(double), 1, fp);
    ret += fread(&model_feature_sample_rate, sizeof(double), 1, fp);
    ret += fread(&model_shrinkage, sizeof(double), 1, fp);
    int mni = 0;
    ret += fread(&mni, sizeof(int), 1, fp);
    if (model_n_iter_cmd) {
      if (model_mode == "test" && model_n_iterations > mni) {
        fprintf(stderr,
                "[WARN] Specified %d iterations. Only %d iterations "
                "found in the model.\n",
                model_n_iterations, mni);
        model_n_iterations = mni;
      }
    } else {
      model_n_iterations = mni;
    }

    ret += fread(&model_save_every, sizeof(int), 1, fp);
    ret += fread(&model_eval_every, sizeof(int), 1, fp);
    loadString(str, fp);
    loadString(model_name, fp);
    loadString(str, fp);
    loadString(model_gap, fp);
    loadString(model_mapping_name, fp);
    
    ret += fread(&abc_sample_rate, sizeof(double), 1, fp);
    ret += fread(&abc_sample_min_data, sizeof(int), 1, fp);
    ret += fread(&warmup_iter, sizeof(int), 1, fp);
    ret += fread(&warmup_use_logit, sizeof(bool), 1, fp);
    
    ret += fread(&regression_l1_loss, sizeof(bool), 1, fp);
    ret += fread(&regression_huber_loss, sizeof(bool), 1, fp);
    ret += fread(&regression_use_hessian, sizeof(bool), 1, fp);
    ret += fread(&regression_auto_clip_value, sizeof(bool), 1, fp);
    ret += fread(&huber_delta, sizeof(double), 1, fp);
    ret += fread(&regression_lp_loss, sizeof(double), 1, fp);

    // Parallelism config
    ret += fread(&use_gpu, sizeof(int), 1, fp);
    ret += fread(&use_omp, sizeof(int), 1, fp);
    ret += fread(&n_threads, sizeof(int), 1, fp);

    // Others
    ret += fread(&save_log, sizeof(bool), 1, fp);
    ret += fread(&save_model, sizeof(bool), 1, fp);
    loadString(str, fp);
    loadString(additional_files, fp);
    ret += fread(&no_map, sizeof(bool), 1, fp);
    ret += fread(&stop_tolerance, sizeof(double), 1, fp);
    ret += fread(&regression_stop_factor, sizeof(double), 1, fp);

    // Rank Query File
    loadString(str, fp);
    loadString(str, fp);

    // ABC related
    ret += fread(&base_candidates_size, sizeof(int), 1, fp);
  }

  void help() {
    printf(
        "ABCBoost usage:\n\
#### Data related:\n\
* `--data_use_mean_as_missing`\n\
* `--data_min_bin_size` minimum size of the bin\n\
* `--data_sparsity_threshold`\n\
* `--data_max_n_bins` max number of bins (default 65535)\n\
* `--data_path, -data` path to train/test data\n\
#### Tree related:\n\
* `--tree_clip_value` gradient clip (default 100)\n\
* `--tree_damping_factor`, regularization on numerator (default 1e-300)\n\
* `--tree_max_n_leaves`, -J (default 4)\n\
* `--tree_min_node_size` (default 10)\n\
#### Model related:\n\
* `--model_use_logit`, whether use logitboost\n\
* `--model_data_sample_rate` (default 1.0)\n\
* `--model_feature_sample_rate` (default 1.0)\n\
* `--model_shrinkage`, `-shrinkage`, `-v`, the learning rate (default 0.08)\n\
* `--model_n_iterations`, `-iter` (default 1000)\n\
* `--model_save_every`, `-save` (default 500)\n\
* `--model_eval_every`, `-eval` (default 1)\n\
* `--model_name`, `-method` regression/lambdarank/mart/abcmart/robustlogit/abcrobustlogit (default abcrobustlogit)\n\
* `--model_pretrained_path`, `-model`\n\
#### Adaptive Base Class (ABC) related:\n\
* `--model_base_candidate_size`, `base_candidates_size`, `-search`, base class searching size in abcmart/abcrobustlogit\n\
* `--model_gap`, `-gap` (default 0) The gap between two base class searchings. For example, `--model_gap 2` means we will do the base class searching in iteration 1, 4, 6, ...\n\
* `--model_warmup_iter`, `-warmup_iter` (default 0) the number of iterations that use normal boosting before ABC method kicks in. It might be helpful for datasets with a large number of classes when we only have a limited base class searching parameter (`-model_base_candidate_size`) \n\
* `--model_warmup_use_logit`, `-warmup_use_logit` 0/1 (default 1) whether use logitboost in warmup iterations.\n\
* `--model_abc_sample_rate`, `-abc_sample_rate` (default 1.0) the sample rate used for the base class searching\n\
* `--model_abc_sample_min_data` `-abc_sample_min_data` (default 2000) the minimum sampled data for base class selection. This parameter only takes into effect when `--abc_sample_rate` is less than `1.0`\n\
#### Regression related:\n\
* `--regression_lp_loss`, `-lp` (default 2.0) whether use Lp norm instead of L2 norm. p (p >= 1.0) has to be specified\n\
* `--regression_use_hessian` 0/1 (default 1) whether use second-order derivatives in the regression. This parameter only takes into effect when `--regression_lp_loss p` is set and `p` is greater than `2`.\n\
* `--regression_huber_loss`, `-huber` 0/1 (default 0) whether use huber loss\n\
* `--regression_huber_delta`, `-huber_delta` the delta parameter for huber loss. This parameter only takes into effect when `--regression_huber_loss 1` is set\n\
#### Parallelism:\n\
* `--n_threads`, `-threads` (default 1)\n\
* `--use_gpu` 0/1 (default 1 if compiled with CUDA) whether use GPU to train models. This parameter only takes into effect when the flag `-DCUDA=on` is set in `cmake`.\n\
#### Other:\n\
* `--save_log`, 0/1 (default 0) whether save the runtime log to file\n\
* `--save_model`, 0/1 (default 1)\n\
* `--no_label`, 0/1 (default 0) It should only be enabled to output prediction file when the testing data has no label and `--model_mode` is `test`\n\
* `--stop_tolerance` (default 1e-13) It works for all non-regression tasks, e.g., classification. The training will stop when the total training loss is less than the stop tolerance.\n\
* `--regression_stop_factor` (default 1e-5) The auto stopping criterion is different from the classification task because the scale of the regression target is unknown. We adaptively set the regression stop tolerate to `regression_stop_factor * total_loss / sum(y^p)`, where `y` is the regression targets and `p` is the value specified in `--regression_lp_loss`.\n\
* `--regression_auto_clip_value` 0/1 (default 1) whether use our adaptive clipping value computation for the predict value on terminal nodes. When enabled, the adaptive clipping value is computed as `tree_clip_value * max_y - min_y` where `tree_clip_value` is set via `--tree_clip_value`, `max_y` and `min_y` are the maximum and minimum regression target value, respectively.\n\
");
  }

  void parseArguments(int argc, char* argv[]) {
    if (argc == 1) {
      help();
    }
    for (int i = 1; i < argc; i += 2) {
      std::string key(argv[i]);
      if (key.length() >= 2 && key[0] == '-') {
        key = (key[1] == '-') ? key.substr(2) : key.substr(1);
      } else {
        fprintf(stderr, "[ERROR] Unrecognize argument [%s]. Did you put '-' for the flags.\n",
                key.c_str());
        help();
        exit(1);
      }
      if(i + 1 >= argc){
        fprintf(stderr, "[ERROR] No value specified for key [%s].\n",argv[i]);
      }
      std::string value = (argv[i + 1]);
      if (key == "iter" || key == "more_iter") model_n_iter_cmd = true;
      parse(key, value);
    }
    if (n_threads > 1)
      use_omp = 1;
    else
      use_omp = 0;
  }

  bool stob(std::string value) const{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
    if(value == "true" || value == "yes" || value == "y")
      return true;
    if(value == "false" || value == "no" || value == "n")
      return false;
    return stoi(value);
  }

  void parse(std::string key, std::string value) {
    try{
      if (key == "data_path" || key == "data") {
        std::string str(value);
        data_path = str;
      } else if (key == "data_is_matrix" || key == "x") {
        data_is_matrix = (stob(value) == 1);
      } else if (key == "data_use_mean_as_missing") {
        data_use_mean_as_missing = (stob(value) == 1);
      } else if (key == "data_max_n_bins") {
        data_max_n_bins = stoi(value);
      } else if (key == "data_min_bin_size") {
        data_min_bin_size = stod(value);
      } else if (key == "data_n_feats") {
        data_n_feats = stoi(value);
      } else if (key == "data_sparsity_threshold") {
        data_sparsity_threshold = stod(value);
      } else
          // tree related
          if (key == "tree_max_n_leaves" || key == "J") {
        tree_max_n_leaves = stoi(value);
      } else if (key == "tree_min_node_size") {
        tree_min_node_size = stoi(value);
      } else if (key == "tree_damping_factor") {
        tree_damping_factor = stod(value);
      } else if (key == "tree_clip_value") {
        tree_clip_value = stod(value);
      } else if (key == "tree_fast_bin_sort") {
        tree_fast_bin_sort = stob(value);
      } else
          // model related
          if (key == "model_n_iterations" || key == "iter") {
        model_n_iterations = stoi(value);
      } else if (key == "more_iter") {
        model_n_iterations = stoi(value);
        model_more_iter = stoi(value);
      } else if (key == "model_shrinkage" || key == "shrinkage" || key == "v") {
        model_shrinkage = stod(value);
      } else if (key == "model_use_logit") {
        model_use_logit = stob(value);
      } else if (key == "model_use_weighted_update") {
        model_use_weighted_update = stob(value);
      } else if (key == "model_is_regression") {
        model_is_regression = stob(value);
      } else if (key == "model_data_sample_rate") {
        model_data_sample_rate = stod(value);
      } else if (key == "model_feature_sample_rate") {
        model_feature_sample_rate = stod(value);
      } else if (key == "model_save_every" || key == "save") {
        model_save_every = stoi(value);
      } else if (key == "model_eval_every" || key == "eval") {
        model_eval_every = stoi(value);
      } else if (key == "model_mode" || key == "mode") {
        std::string str(value);
        model_mode = str;
      } else if (key == "model_gap" || key == "gap") {
        std::string str(value);
        model_gap = str;
      } else if (key == "model_name" || key == "method") {
        std::string str(value);
        model_name = str;
      } else if (key == "model_pretrained_path" || key == "model") {
        std::string str(value);
        model_pretrained_path = str;
      } else
          // others
          if (key == "experiment_folder") {
        std::string str(value);
        experiment_folder = str;
      } else if (key == "n_threads" || key == "threads") {
        n_threads = stoi(value);
        if(n_threads > 1)
          use_omp = 1;
        else
          use_omp = 0;
      } else if (key == "save_log") {
        save_log = stob(value) ;
      } else if (key == "save_model") {
        save_model = stob(value);
      } else if (key == "use_gpu") {
        use_gpu = stob(value);
      } else if (key == "use_omp") {
        use_omp = stob(value);
      } else if (key == "additional_files") {
        additional_files = std::string(value);
      } else if (key == "no_map") {
        no_map = stob(value);
      } else if (key == "no_label") {
        no_label = stob(value);
      } else if (key == "stop_tolerance") {
        stop_tolerance = stod(value);
      } else if (key == "regression_stop_factor") {
        regression_stop_factor = stod(value);
      } else if (key == "model_warmup_iter" || key == "warmup_iter") {
        warmup_iter = stoi(value);
      } else if (key == "model_warmup_use_logit" || key == "warmup_use_logit") {
        warmup_use_logit = stob(value);
      } else if (key == "regression_l1_loss" || key == "l1") {
        regression_l1_loss = stob(value);
      } else if (key == "regression_huber_loss" || key == "huber") {
        regression_huber_loss = stob(value);
      } else if (key == "regression_huber_delta" || key == "huber_delta") {
        huber_delta = stod(value);
      } else if (key == "regression_use_hessian") {
        regression_use_hessian = stob(value);
      } else if (key == "regression_auto_clip_value") {
        regression_auto_clip_value = stob(value);
      } else if (key == "regression_lp_loss" || key == "lp") {
        regression_lp_loss = stod(value);
        if(regression_lp_loss == 1.0)
          regression_l1_loss = true;
        if(regression_lp_loss < 1){
          printf("[ERROR] Unsupported Lp value [%f] (p must be at least 1).\n",regression_lp_loss);
          throw "Unsupported argument exception\n";
        }
      } else if (key == "rank_query_file" || key == "query") {
        rank_query_file = std::string(value);
      } else if (key == "prediction_file") {
        prediction_file = std::string(value);
      } else if (key == "model_base_candidate_size" || key == "base_candidates_size" || key == "search") {
        base_candidates_size = stoi(value);
      } else if (key == "model_abc_sample_rate" || key == "abc_sample_rate") {
        abc_sample_rate = stod(value);
      } else if (key == "model_abc_sample_min_data" || key == "abc_sample_min_data") {
        abc_sample_min_data = stoi(value);
      } else {
        printf("[ERROR] Unknown argument [%s]\n", key.c_str());
        if(from_wrapper == false){
          help();
          exit(1);
        }else{
          throw "Unknown argument exception\n";
        }
      }
    }catch(const std::invalid_argument& e){
      fprintf(stderr,"[ERROR] Failed to parse [%s] for argument [%s]\n",value.c_str(),key.c_str());
    }
  }

  void sanityCheck() {
    // TODO(@weijie): add a sanity check for the arguments.
  }

  std::string getDataName() {
    int data_name_start_nix = data_path.find_last_of('/') + 1;
    int data_name_start_win = data_path.find_last_of('\\') + 1;
    int data_name_start = std::max(data_name_start_nix, data_name_start_win);
    std::string data_name = data_path.substr(data_name_start);
    return data_name;
  }

  std::string getMappingName() {
    return getDataName() + mapping_suffix; 
  }
};

}  // namespace ABCBoost

#endif  // ABCBOOST_CONFIG_H
