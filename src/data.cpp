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

#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <list>

#include "config.h"
#include "data.h"
#ifdef OMP
#include "omp.h"
#else
#include "dummy_omp.h"
#endif
#include "utils.h"

namespace ABCBoost {

/**
 * Constructor of Data class.
 */
Data::Data(Config* config) {
  this->config = config;
  data_header.n_feats = (config->data_is_matrix) ? 0 : config->data_n_feats;
  n_data = data_header.n_classes = 0;
}

/**
 * Destructor of Data class.
 */
Data::~Data() {}

void Data::loadRankQuery() {
  FILE* fp = fopen(config->rank_query_file.c_str(), "r");
  if (fp == NULL) {
    fprintf(stderr,
            "[Error] Cannot open rank query file located at (%s). Please check "
            "the path!\n",
            config->rank_query_file.c_str());
    exit(1);
  }
  size_t last = 0;
  size_t len = 0;
  this->rank_groups.clear();
  while (fscanf(fp, "%zu", &len) != EOF) {
    std::pair<size_t, size_t> p;
    p.first = last;
    p.second = last + len;
    this->rank_groups.push_back(p);
    last = p.second;
  }
  //  if(last != data->n_data){
  //    fprintf(stderr,"[Error] Lambda query file content does not match the
  //    data. Found (%zu) in query file while (%u) in
  //    data!\n",last,data->n_data); exit(1);
  //  }
  fclose(fp);
}

/**
 * Load data from the data path.
 * @param[in] fileptr: Pointer to the FILE object
 */
void Data::loadData(bool from_scratch) {
  if (!doesFileExist(config->data_path) && config->from_wrapper == false) {
    fprintf(stderr, "[ERROR] Data file does not exist!\n");
    exit(1);
  }
  Utils::Timer timer = Utils::Timer();
  Utils::Timer timer_all = Utils::Timer();
  double t1, t2, t3, t4, t5, t6;
  printf("Loading data ...\n");
  timer.restart();
  timer_all.restart();

  // reading data from file
  if (config->from_wrapper == true) {
    if(config->mem_is_sparse){
      loadMemoryKeyValueFormat(config->mem_Y_matrix,config->mem_X_kv,
                               config->mem_n_row,config->mem_n_col);
    }else{
      loadMemoryColumnMajorMatrix(config->mem_Y_matrix, config->mem_X_matrix,
                                  config->mem_n_row, config->mem_n_col);
    }
  } else {
    config->data_is_matrix = testDataIsMatrix(config->data_path);
    if (config->data_is_matrix) {
      loadMatrixFormat(config->data_path);
    } else {
      loadLibsvmFormat(config->data_path);
    }
  }

  n_data = Y.size();

  printf("-- finish reading data in %.4f seconds.\n", timer.get_time_restart());

  featureCleanUp();
  printf("-- finish cleaning up empty features in %.4f seconds.\n",
         timer.get_time_restart());
	
	bool is_created_quantization = false;

	if (config->no_map == true){
		load();
    printf("-- skip adaptive quantization, using truncated raw data.\n");
	}else if (from_scratch == false) {
    load();
    printf("-- finish loading adaptive quantization in %.4f seconds.\n",
           timer.get_time_restart());
  } else if (config->model_mode == "train") {
    adaptiveQuantization();  // discretize the features using adaptive bin sizes
    printf("-- finish creating adaptive quantization in %.4f seconds.\n",
           timer.get_time_restart());
		is_created_quantization = true;
  } else if (config->from_wrapper) {
    load();
  } else {
    printf("[Error] No .map file found in test!\n");
    exit(1);
  }

  restoreDenseFeatures();
  printf("-- finish restoring dense features in %.4f seconds.\n",
         timer.get_time_restart());
  // normalize labels to be consecutive integers
  if (config->model_name != "regression" && config->model_name != "lambdamart" && config->model_name != "gbrank")
    normalizeLabels();
  else
    data_header.n_classes = 1;

  printf("-- finish label normalization in %.4f seconds.\n",
         timer.get_time_restart());

  printf("Finish loading data in %.4f seconds.\n", timer_all.get_time());
  printSummary();

  std::string mapping_name = config->getMappingName();
}

/**
 * Print out the data, for debugging or visualization.
 */
void Data::printData(unsigned int n) {
  printf("Showing the first %d rows of data.\n", n);
  std::vector<unsigned int> indices =
      std::vector<unsigned int>(data_header.n_feats, 0);
  int val = 0;
  for (int i = 0; i < n; ++i) {
    printf("%5.0f |", Y[i]);
    for (int j : valid_fi) {
      if (dense_f[j]) {
        val = Xv[j][i];
      } else {
        val = (Xi[j][indices[j]] == i) ? Xv[j][indices[j]]
                                       : data_header.unobserved_fv[j];
        if (Xi[j][indices[j]] == i) ++indices[j];
      }
      printf("%3d ", val);
    }
    printf("\n");
  }
}

/**
 * Print out summary of data.
 */
void Data::printSummary() {
  printf(
      "\nData summary: | # data: %d "
      "| # features: %d # splittable: %zu (# dense: %d) | # classes: %d |\n",
      n_data, data_header.n_feats, valid_fi.size(), n_dense, data_header.n_classes);
}

/**
 * Save data information to file.
 * @param[in] fileptr: Pointer to the FILE object
 */
void Data::saveData(FILE* fileptr) { data_header.serialize(fileptr); }

void Data::dumpData(FILE* fileptr, std::string format){
  if(format == "libsvm"){
    dumpLibsvm(fileptr);
  }else if(format == "csv"){
    dumpCSV(fileptr);
  }else{
    printf("[ERROR] Unsuported dump format (%s), which must be libsvm or csv\n",format.c_str());
  }
} 

void Data::dumpLibsvm(FILE* fileptr) {
  const auto n = Y.size();
  const bool data_remap = data_header.idx2label.size() > 0;
  std::vector<unsigned int> indices =
      std::vector<unsigned int>(data_header.n_feats, 0);
  int val = 0;
  for (int i = 0; i < n; ++i) {
    fprintf(fileptr,"%g",data_remap ? data_header.idx2label[Y[i]] : Y[i]);
    for (int j : valid_fi) {
      if (dense_f[j]) {
        val = Xv[j][i];
      } else {
        val = (Xi[j][indices[j]] == i) ? Xv[j][indices[j]] : 0;
        if (Xi[j][indices[j]] == i) ++indices[j];
      }
      if(val != 0){
        fprintf(fileptr," %d:%d", j + 1, val);
      }
    }
    fprintf(fileptr,"\n");
  }
}

void Data::dumpCSV(FILE* fileptr) {
  const auto n = Y.size();
  const bool data_remap = data_header.idx2label.size() > 0;
  const int n_feats = data_header.n_feats;
  std::vector<unsigned int> indices =
      std::vector<unsigned int>(n_feats, 0);
  int val = 0;
  for (int i = 0; i < n; ++i) {
    fprintf(fileptr,"%g",data_remap ? data_header.idx2label[Y[i]] : Y[i]);
    for (int j = 0;j < n_feats;++j){
      if (Xv[j].size() == 0){
        val = 0;
      } else if (dense_f[j]) {
        val = Xv[j][i];
      } else {
        val = (Xi[j][indices[j]] == i) ? Xv[j][indices[j]] : 0;
        if (Xi[j][indices[j]] == i) ++indices[j];
      }
      fprintf(fileptr,",%d", val);
    }
    fprintf(fileptr,"\n");
  }
}

void Data::loadDataHeader(FILE* fileptr) {
  fread(&data_header.n_feats, sizeof(unsigned int), 1, fileptr);
  fread(&data_header.n_classes, sizeof(unsigned int), 1, fileptr);
  data_header.unobserved_fv.resize(data_header.n_feats);
  for (int i = 0; i < data_header.n_feats; ++i) {
    fread(&data_header.unobserved_fv[i], sizeof(unsigned short), 1, fileptr);
  }
  data_header.n_bins_per_f.resize(data_header.n_feats);
  data_header.bin_starts_per_f.resize(data_header.n_feats);
  for (int i = 0; i < data_header.n_feats; ++i) {
    fread(&data_header.n_bins_per_f[i], sizeof(int), 1, fileptr);
    data_header.bin_starts_per_f[i].resize(data_header.n_bins_per_f[i]);
    for (int j = 0; j < data_header.n_bins_per_f[i]; ++j) {
      fread(&data_header.bin_starts_per_f[i][j], sizeof(double), 1, fileptr);
    }
  }
  if (!config->model_is_regression) {
    data_header.idx2label.resize(data_header.n_classes);
    for (int i = 0; i < data_header.n_classes; ++i) {
      fread(&data_header.idx2label[i], sizeof(double), 1, fileptr);
    }
  }
}

std::vector<std::string> Data::split(const std::string& s, char delimiter) {
  std::vector<std::string> ret;
  std::string now = "";
  for (const auto& ch : s) {
    if (ch == delimiter) {
      ret.push_back(now);
      now = "";
    } else {
      now += ch;
    }
  }
  if (now != "") ret.push_back(now);
  return ret;
}


/**
 * Do the adaptive quantization for each feature.
 * Details are from P Li et al., 2008.
 */
void Data::adaptiveQuantization() {
  data_header.bin_starts_per_f.resize(data_header.n_feats);
  data_header.n_bins_per_f.resize(data_header.n_feats, 0);
  data_header.unobserved_fv.resize(data_header.n_feats, 0);

  // initialize the 2D vectors of values
  Xv.resize(data_header.n_feats);
  for (int j = 0; j < data_header.n_feats; ++j) {
    Xv[j].resize(Xv_raw[j].size());
  }
  config->default_label = data_header.idx2label.size() > 0 ? data_header.idx2label[0] : Y[0];

  std::vector<std::string> additional_files = split(config->additional_files);
  std::vector<std::unique_ptr<Data>> data_pool;
  // std::vector<Data*> data_pool;
  for (const auto& file : additional_files) {
    data_pool.emplace_back(new Data(config));
  }
  for (int i = 0; i < data_pool.size(); ++i) {
    if (testDataIsMatrix(additional_files[i])) {
      data_pool[i]->loadMatrixFormat(additional_files[i]);
    } else {
      data_pool[i]->loadLibsvmFormat(additional_files[i]);
    }
  }
  std::vector<std::string> additional_files_no_label = split(config->additional_files_no_label);
  auto before_size = data_pool.size();
  for (const auto& file : additional_files_no_label) {
    data_pool.emplace_back(new Data(config));
  }

  auto prev_no_label_config = config->no_label;
  config->no_label = true;
  for (int i = 0; i < additional_files_no_label.size(); ++i) {
    if (testDataIsMatrix(additional_files_no_label[i])) {
      data_pool[before_size + i]->loadMatrixFormat(additional_files_no_label[i]);
    } else {
      data_pool[before_size + i]->loadLibsvmFormat(additional_files_no_label[i]);
    }
  }
  config->no_label = prev_no_label_config;

#pragma omp parallel for
  for (int f = 0; f < valid_fi.size(); ++f) {
    auto j = valid_fi[f];
    std::vector<double> fv = Xv_raw[j];
    for (const auto& pdata : data_pool) {
      if (j >= pdata->Xv_raw.size()) break;
      fv.insert(fv.end(), pdata->Xv_raw[j].begin(), pdata->Xv_raw[j].end());
    }
    double unobserved = (config->data_use_mean_as_missing)
                            ? std::accumulate(fv.begin(), fv.end(), 0) * 1.0 / fv.size()
                            : 0.0;
		fv.push_back(unobserved);
    unsigned int n_j = fv.size();

    std::sort(fv.begin(), fv.end());
    double bin_size = config->data_min_bin_size;
    std::vector<double> bin_starts;
    while (true) {  // determine the bin size
      unsigned int idx = 0;
      while (idx < n_j) {
        double bin_start = fv[idx];
        double bin_end = bin_start + bin_size;
        bin_starts.push_back(bin_start);

        if (bin_starts.size() > config->data_max_n_bins) break;
        while (idx < n_j && fv[idx] <= bin_end) idx++;
      }
      if (idx == n_j) {
        break;  // successfully quantize the data
      } else {
        bin_starts.clear();
        bin_size *= 2;
      }
    }

    // after finding the quantization points, start to discretize
    // the values of the current feature of the entire data set.
    std::vector<double>& xvr = Xv_raw[j];
    std::vector<data_quantized_t>& xv = Xv[j];
    for (int i = 0; i < xvr.size(); ++i) {
      xv[i] = Data::discretize(bin_starts, xvr[i]);
    }
    data_header.n_bins_per_f[j] = bin_starts.size();
    data_header.unobserved_fv[j] = Data::discretize(bin_starts, unobserved);
    data_header.bin_starts_per_f[j] = bin_starts;
  }
  std::vector<std::vector<double>>().swap(Xv_raw);  // clear Xv_raw
}

/**
 * Helper method to discretize the value according
 * to bin quantization points.
 * @param bin_starts the bin quantization points
 *        value the current value
 * @return the discretized value
 */
unsigned short Data::discretize(std::vector<double>& bin_starts, double v) {
  unsigned short l = 0, r = bin_starts.size(), m;
  while (r - l > 1) {
    m = (r + l) / 2;
    if (bin_starts[m] > v)
      r = m;
    else
      l = m;
  }
  if (l + 1 < bin_starts.size() && bin_starts[l + 1] - v < v - bin_starts[l]) {
    return l + 1;
  } else {
    return l;
  }
}

/**
 * Helper method to determine if a file exists.
 * @param path the file path.
 * @return true if it exists.
 */
bool Data::doesFileExist(std::string path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

/**
 * Helper method to store non empty features.
 */
void Data::featureCleanUp() {
  valid_fi.clear();
	if(config->model_mode == "train"){
		for (unsigned int j = 0; j < Xi.size(); ++j) {
	//    if (!Xv_raw[j].empty()) valid_fi.push_back(j);
			if (Xv_raw[j].size() >= 2 * config->tree_min_node_size) valid_fi.push_back(j);
		}
	}else{
		for (unsigned int j = 0; j < Xi.size(); ++j) {
	    if (!Xv_raw[j].empty()) valid_fi.push_back(j);
		}
	}
}

/**
 * Helper method to load pretrained data information.
 * @param[in] fp: Pointer to the FILE object
 */
void Data::load() {
  // load data information from file

  size_t ret = 0;
	if (config->no_map == true){
		if(config->model_mode == "train"){
			data_header.n_bins_per_f.resize(data_header.n_feats,0);
			data_header.unobserved_fv.resize(data_header.n_feats);
			for(int j = 0;j < data_header.n_feats;++j)
				data_header.unobserved_fv[j] = (config->data_use_mean_as_missing) ? std::accumulate(Xv_raw[j].begin(), Xv_raw[j].end(), 0) * 1.0 / Xv_raw[j].size() : 0.0;
		}else{
			//;
		}
	} else if (config->from_wrapper == true) {
    Xv.clear();
    data_header.n_feats = config->mem_n_col;
  } else {
    //;
  }
  Xv_raw.resize(data_header.n_feats);
  Xi.resize(data_header.n_feats);
  auxData.resize(data_header.n_feats);
  auxDataWidth.resize(data_header.n_feats);
  while (valid_fi.back() >= data_header.n_feats) valid_fi.pop_back();

  // initialize the 2D vectors of values
  Xv.resize(data_header.n_feats);
  for (int j = 0; j < data_header.n_feats; ++j) {
    Xv[j].resize(Xv_raw[j].size());
    for (unsigned int i = 0; i < Xv_raw[j].size(); ++i) {
			if(config->no_map == true){
				if(Xv_raw[j][i] < 0){
    			fprintf(stderr,"[Error] Values must be non-negative in no_map mode. %f detected\n", Xv_raw[j][i]);
					exit(1);
				}
				Xv[j][i] = Xv_raw[j][i];
				if(data_header.n_bins_per_f[j] < Xv[j][i] + 1)
					data_header.n_bins_per_f[j] = Xv[j][i] + 1;
			}else{
	      Xv[j][i] =
  	        Data::discretize(data_header.bin_starts_per_f[j], Xv_raw[j][i]);
			}
    }
  }
}

void Data::loadMemoryKeyValueFormat(const double* Y_matrix, const std::vector<std::vector<std::pair<int,double>>>& X_kv,int n_row, int n_col) {
  int T = 1;
#pragma omp parallel
#pragma omp master
  {
    T = config->n_threads;  // omp_get_num_threads();
  }
  omp_set_num_threads(T);
  int n_lines = n_row;
  int step = (n_lines + T - 1) / T;

  std::vector<std::vector<std::vector<unsigned int>>> i_global;
  std::vector<std::vector<std::vector<double>>> v_global;
  std::vector<std::vector<double>> Y_global;
  std::vector<unsigned int> n_feat_global;
  i_global.resize(T);
  v_global.resize(T);
  Y_global.resize(T);
  n_feat_global.resize(T);

  omp_set_num_threads(T);
#pragma omp parallel
  {
    int n_feats_local = 0;
    int t = omp_get_thread_num();
    int start = t * step, end = std::min(n_lines, (t + 1) * step);

    for (int i = start; i < end; ++i) {
			if(config->no_label == false){
      	Y_global[t].push_back(Y_matrix[i]);
			}else{
      	Y_global[t].push_back(config->default_label);
			}
      int j;
      double j_val;
      bool is_key = true;
      for(int k = 0;k < X_kv[i].size();++k){
        j = X_kv[i][k].first;
        j_val = X_kv[i][k].second;
        if (config->data_n_feats < 1 && n_feats_local < j) {
          n_feats_local = j;
          i_global[t].resize(n_feats_local);
          v_global[t].resize(n_feats_local);
        }
        i_global[t][j - 1].push_back(i);
        v_global[t][j - 1].push_back(j_val);
      }
    }

    n_feat_global[t] = n_feats_local;

#pragma omp critical
    {
      if (n_feats_local > data_header.n_feats)
        data_header.n_feats = n_feats_local;
    };
  }
  
  Xi.clear();
  Xv_raw.clear();
  Y.clear();
  Xv.clear();

  Xi.resize(data_header.n_feats);
  Xv_raw.resize(data_header.n_feats);
  auxData.resize(data_header.n_feats);
  auxDataWidth.resize(data_header.n_feats);
  for (int t = 0; t < T; t++) {
#pragma omp parallel for
    for (int j = 0; j < n_feat_global[t]; ++j) {
      Xi[j].insert(Xi[j].end(), i_global[t][j].begin(), i_global[t][j].end());
      Xv_raw[j].insert(Xv_raw[j].end(), v_global[t][j].begin(),
                       v_global[t][j].end());
    }
    Y.insert(Y.end(), Y_global[t].begin(), Y_global[t].end());
  }
  printf("loading done\n");
}

/**
 * Load data from the data path.
 * @param path the data path.
 *        is_train whether it is train data.
 */
void Data::loadMemoryColumnMajorMatrix(double* Y_matrix, double* X_matrix,
                                       int n_row, int n_col) {
  int T = 1;
#pragma omp parallel
#pragma omp master
  {
    T = config->n_threads;  // omp_get_num_threads();
  }
  omp_set_num_threads(T);
  int n_lines = n_row;
  int step = (n_lines + T - 1) / T;

  std::vector<std::vector<std::vector<unsigned int>>> i_global;
  std::vector<std::vector<std::vector<double>>> v_global;
  std::vector<std::vector<double>> Y_global;
  std::vector<unsigned int> n_feat_global;
  i_global.resize(T);
  v_global.resize(T);
  Y_global.resize(T);
  n_feat_global.resize(T);

  omp_set_num_threads(T);
  data_header.n_feats = n_col;
#pragma omp parallel
  {
    int n_feats_local = n_col;
    int t = omp_get_thread_num();
    i_global[t].resize(n_col);
    v_global[t].resize(n_col);
    int start = t * step, end = std::min(n_lines, (t + 1) * step);

    for (int i = start; i < end; ++i) {
      Y_global[t].push_back(Y_matrix[i]);
      for (int j = 0; j < n_col; ++j) {
        double j_val = X_matrix[j * n_row + i];
        if (j_val != 0) {
          i_global[t][j].push_back(i);
          v_global[t][j].push_back(j_val);
        }
      }
    }

    n_feat_global[t] = n_feats_local;

#pragma omp critical
    {
      if (n_feats_local > data_header.n_feats)
        data_header.n_feats = n_feats_local;
    };
  }

  Xi.clear();
  Xv_raw.clear();
  Y.clear();
  Xv.clear();

  Xi.resize(data_header.n_feats);
  Xv_raw.resize(data_header.n_feats);
  auxData.resize(data_header.n_feats);
  auxDataWidth.resize(data_header.n_feats);
  for (int t = 0; t < T; t++) {
#pragma omp parallel for
    for (int j = 0; j < n_feat_global[t]; ++j) {
      Xi[j].insert(Xi[j].end(), i_global[t][j].begin(), i_global[t][j].end());
      Xv_raw[j].insert(Xv_raw[j].end(), v_global[t][j].begin(),
                       v_global[t][j].end());
    }
    Y.insert(Y.end(), Y_global[t].begin(), Y_global[t].end());
  }
  printf("loading done\n");
}

/**
 * Load data from the data path.
 * @param path the data path.
 *        is_train whether it is train data.
 */
void Data::loadMatrixFormat(std::string path) {
  const char* delimiter = " ,\t";
  std::ifstream infile(path);
  std::string line;
  std::vector<std::string> buffer;

  while (getline(infile, line)) {
    buffer.push_back(line);
  }

  int T = 1;
#pragma omp parallel
#pragma omp master
  {
    T = config->n_threads;  // omp_get_num_threads();
  }
  omp_set_num_threads(T);
  int n_lines = buffer.size();
  int step = (n_lines + T - 1) / T;

  std::vector<std::vector<std::vector<unsigned int>>> i_global;
  std::vector<std::vector<std::vector<double>>> v_global;
  std::vector<std::vector<double>> Y_global;
  std::vector<unsigned int> n_feat_global;
  i_global.resize(T);
  v_global.resize(T);
  Y_global.resize(T);
  n_feat_global.resize(T);

  omp_set_num_threads(T);

#pragma omp parallel
  {
    int n_feats_local = 0;
    int t = omp_get_thread_num();
    int start = t * step, end = std::min(n_lines, (t + 1) * step);

    for (int i = start; i < end; ++i) {
      char* token = &(buffer[i][0]);
      char* ptr;
      char* pos = NULL;
			if(config->no_label == false){
      	pos = strtok_r(token, delimiter, &ptr);
      	Y_global[t].push_back(atof(pos));
			}else{
      	Y_global[t].push_back(config->default_label);
			}
      int j = 1;
      while (true) {
				if(config->no_label && pos == NULL)
					pos = strtok_r(token, delimiter, &ptr);
				else
	        pos = strtok_r(NULL, delimiter, &ptr);
        if (pos == NULL) break;
        double j_val = atof(pos);
        if (config->data_n_feats < 1 && n_feats_local < j) {
          n_feats_local = j;
          i_global[t].resize(n_feats_local);
          v_global[t].resize(n_feats_local);
        }
        if (j_val != 0) {
          i_global[t][j - 1].push_back(i);
          v_global[t][j - 1].push_back(j_val);
        }
        j++;
      }
    }

    n_feat_global[t] = n_feats_local;

#pragma omp critical
    {
      if (n_feats_local > data_header.n_feats)
        data_header.n_feats = n_feats_local;
    };
  }

  Xi.resize(data_header.n_feats);
  Xv_raw.resize(data_header.n_feats);
  auxData.resize(data_header.n_feats);
  auxDataWidth.resize(data_header.n_feats);
  for (int t = 0; t < T; t++) {
#pragma omp parallel for
    for (int j = 0; j < n_feat_global[t]; ++j) {
      Xi[j].insert(Xi[j].end(), i_global[t][j].begin(), i_global[t][j].end());
      Xv_raw[j].insert(Xv_raw[j].end(), v_global[t][j].begin(),
                       v_global[t][j].end());
    }
    Y.insert(Y.end(), Y_global[t].begin(), Y_global[t].end());
  }
}

/**
 * Load data from the data path.
 * @param path the data path.
 *        is_train whether it is train data.
 */
void Data::loadLibsvmFormat(std::string path) {
  const char* delimiter = " ,:";
  std::ifstream infile(path);
  std::string line;
  std::vector<std::string> buffer;

  while (getline(infile, line)) {
    buffer.push_back(line);
  }

  int T = 1;
#pragma omp parallel
#pragma omp master
  {
    T = config->n_threads;  // omp_get_num_threads();
  }
  omp_set_num_threads(T);
  int n_lines = buffer.size();
  int step = (n_lines + T - 1) / T;

  std::vector<std::vector<std::vector<unsigned int>>> i_global;
  std::vector<std::vector<std::vector<double>>> v_global;
  std::vector<std::vector<double>> Y_global;
  std::vector<unsigned int> n_feat_global;
  i_global.resize(T);
  v_global.resize(T);
  Y_global.resize(T);
  n_feat_global.resize(T);

  omp_set_num_threads(T);
#pragma omp parallel
  {
    int n_feats_local = 0;
    int t = omp_get_thread_num();
    int start = t * step, end = std::min(n_lines, (t + 1) * step);

    for (int i = start; i < end; ++i) {
      char* token = &(buffer[i][0]);
      char* ptr;
      char* pos = NULL;
			if(config->no_label == false){
				pos = strtok_r(token, delimiter, &ptr);
      	Y_global[t].push_back(atof(pos));
			}else{
      	Y_global[t].push_back(config->default_label);
			}
      int j;
      double j_val;
      bool is_key = true;
      while (true) {
				if(config->no_label && pos == NULL)
					pos = strtok_r(token, delimiter, &ptr);
				else
	        pos = strtok_r(NULL, delimiter, &ptr);
        if (pos == NULL) break;
        if (is_key) {
          is_key = false;
          j = atoi(pos);
        } else {
          is_key = true;
          j_val = atof(pos);
          if (config->data_n_feats < 1 && n_feats_local < j) {
            n_feats_local = j;
            i_global[t].resize(n_feats_local);
            v_global[t].resize(n_feats_local);
          }
          i_global[t][j - 1].push_back(i);
          v_global[t][j - 1].push_back(j_val);
        }
      }
    }

    n_feat_global[t] = n_feats_local;

#pragma omp critical
    {
      if (n_feats_local > data_header.n_feats)
        data_header.n_feats = n_feats_local;
    };
  }

  Xi.resize(data_header.n_feats);
  Xv_raw.resize(data_header.n_feats);
  auxData.resize(data_header.n_feats);
  auxDataWidth.resize(data_header.n_feats);
  for (int t = 0; t < T; t++) {
#pragma omp parallel for
    for (int j = 0; j < n_feat_global[t]; ++j) {
      Xi[j].insert(Xi[j].end(), i_global[t][j].begin(), i_global[t][j].end());
      Xv_raw[j].insert(Xv_raw[j].end(), v_global[t][j].begin(),
                       v_global[t][j].end());
    }
    Y.insert(Y.end(), Y_global[t].begin(), Y_global[t].end());
  }
}

/**
 * Normalize the labels to consecutive discrete numbers;
 */
void Data::normalizeLabels() {
  unsigned int idx = data_header.n_classes;
  for (unsigned int i = 0; i < n_data; ++i) {
    double y = Y[i];
    if (data_header.label2idx.find(y) == data_header.label2idx.end()) {
      data_header.label2idx[y] = idx;
      data_header.idx2label.push_back(y);
      Y[i] = idx++;
    } else {
      Y[i] = data_header.label2idx[y];
    }
  }
  data_header.n_classes = idx;
}

/**
 * Restore a feature representation to dense format if the frequency of that
 * feature exceeds the sparse threshold.
 */
void Data::restoreDenseFeatures() {
  dense_f = std::vector<short>(data_header.n_feats, 0);
  i_offset = std::vector<unsigned int>(data_header.n_feats + 1, 0);
  v_offset = std::vector<unsigned int>(data_header.n_feats + 1, 0);
  n_dense = 0;
  unsigned int i_sz = 0, v_sz = 0, sz;

  for (int j = 0; j < data_header.n_feats; ++j) {
    i_offset[j] = i_sz;
    v_offset[j] = v_sz;
    sz = Xi[j].size();
    if (1.0 * sz / n_data > config->data_sparsity_threshold) {
      dense_f[j] = 1;
      n_dense++;
      std::vector<data_quantized_t> tmp(n_data, data_header.unobserved_fv[j]);
      for (int i = 0; i < sz; ++i) tmp[Xi[j][i]] = Xv[j][i];
      Xv[j] = tmp;
      Xi[j].clear();
      Xi[j].shrink_to_fit();
      v_sz += n_data;
    } else {
      i_sz += sz;
      v_sz += sz;
    }
  }
  i_offset[data_header.n_feats] = i_sz;
  v_offset[data_header.n_feats] = v_sz;
  this->i_sz = i_sz;
  this->v_sz = v_sz;
}

bool Data::testDataIsMatrix(std::string path) {
  FILE* fp = fopen(path.c_str(), "r");
  bool is_matrix = true;
  const int TEST_CHAR_LENGTH = 100;
  for (int i = 0; i < TEST_CHAR_LENGTH; ++i) {
    char ch;
    if (fscanf(fp, "%c", &ch) == EOF) break;
    if (ch == ':') {
      is_matrix = false;
      break;
    }
  }
  fclose(fp);
  return is_matrix;
}

void Data::constructAuxData(){
	for(int i = 0;i < valid_fi.size();++i){
		int fid = valid_fi[i];
		if(data_header.n_bins_per_f[fid] <= 16){
			auxDataWidth[fid] = 1;
			const auto &fv = Xv[fid];
			if(dense_f[fid]){
				auxData[fid] = std::vector<uint8_t>((n_data + 1) / 2);
				for(int i = 0;i < n_data;++i)
					if((i & 1) == 0)
						auxData[fid][i >> 1] |= fv[i];
					else
						auxData[fid][i >> 1] |= fv[i] << 4;
			}else{
				const auto &fi = Xi[fid];
				auxData[fid] = std::vector<uint8_t>((fi.size() + 1) / 2);
				for(int i = 0;i < fi.size();++i)
					if((i & 1) == 0)
						auxData[fid][i >> 1] |= fv[i];
					else
						auxData[fid][i >> 1] |= fv[i] << 4;
			}
		}
	}
}

}  // namespace ABCBoost
