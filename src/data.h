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

#ifndef ABCBOOST_DATA_H
#define ABCBOOST_DATA_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "utils.h"

namespace ABCBoost {

#ifndef CUDA
typedef unsigned short data_quantized_t;
#else
typedef unsigned short data_quantized_t;
#endif
//typedef uint8_t data_quantized_t;

struct DataHeader {
  unsigned int n_feats;
  unsigned int n_classes;
  std::vector<unsigned short> unobserved_fv;  // unobserved values for features
  std::vector<int> n_bins_per_f;  // number of discrete categories per feature
  std::vector<std::vector<double>> bin_starts_per_f;
  std::vector<double> idx2label;  // inverse mapping of label2idx
  std::unordered_map<double, unsigned short> label2idx;  // reordered labels

  void serialize(FILE* fp) {
    Utils::serialize(fp, n_feats);
    Utils::serialize(fp, n_classes);
    Utils::serialize_vector(fp, unobserved_fv);
    Utils::serialize_vector(fp, n_bins_per_f);
    Utils::serialize_vector2d(fp, bin_starts_per_f);
    Utils::serialize_vector(fp, idx2label);
  }
  
	void serialize_no_map(FILE* fp) {
    Utils::serialize(fp, n_feats);
    Utils::serialize(fp, n_classes);
    Utils::serialize_vector(fp, unobserved_fv);
    Utils::serialize_vector(fp, n_bins_per_f);
    Utils::serialize_vector(fp, idx2label);
  }

  static DataHeader deserialize(FILE* fp) {
    DataHeader data_header;
    data_header.n_feats = Utils::deserialize<unsigned int>(fp);
    data_header.n_classes = Utils::deserialize<unsigned int>(fp);
    data_header.unobserved_fv = Utils::deserialize_vector<unsigned short>(fp);
    data_header.n_bins_per_f = Utils::deserialize_vector<int>(fp);
    data_header.bin_starts_per_f = Utils::deserialize_vector2d<double>(fp);
    data_header.idx2label = Utils::deserialize_vector<double>(fp);
    data_header.label2idx.clear();
    for (int i = 0; i < data_header.idx2label.size(); ++i) {
      data_header.label2idx[data_header.idx2label[i]] = i;
    }
    return data_header;
  }
  
	static DataHeader deserialize_no_map(FILE* fp) {
    DataHeader data_header;
    data_header.n_feats = Utils::deserialize<unsigned int>(fp);
    data_header.n_classes = Utils::deserialize<unsigned int>(fp);
    data_header.unobserved_fv = Utils::deserialize_vector<unsigned short>(fp);
    data_header.n_bins_per_f = Utils::deserialize_vector<int>(fp);
    data_header.idx2label = Utils::deserialize_vector<double>(fp);
    data_header.label2idx.clear();
    for (int i = 0; i < data_header.idx2label.size(); ++i) {
      data_header.label2idx[data_header.idx2label[i]] = i;
    }
    return data_header;
  }
};

class Data {
 public:
  DataHeader data_header;

  std::vector<std::vector<double>> Xv_raw;      // the corresponding raw values
  std::vector<std::vector<unsigned int>> Xi;    // store the instance ids
  std::vector<std::vector<data_quantized_t>> Xv;  // store the corresponding value
  std::vector<std::vector<data_quantized_t>> Xv_tmp;  // store the corresponding value
  std::vector<double> Y;                        // normalized label values
  std::vector<short> dense_f;  // whether features are dense or sparse
  std::vector<unsigned int> i_offset, v_offset;  // the offset used in GPU
  std::vector<unsigned int> valid_fi;            // non-empty feature indices

  double sparsity_threshold;  // above which we will treat the feature as dense
  unsigned int n_data, n_dense;

	int n_sample_feats;

  unsigned int i_sz, v_sz;  // total size of Xi, Xv, for GPU usage
  std::vector<std::pair<size_t, size_t>> rank_groups;

  Config* config;

	std::vector<int> auxDataWidth;
	std::vector<std::vector<uint8_t>> auxData;

  Data(Config* config);
  ~Data();

  void load(FILE* fileptr);
  void loadData(FILE* fileptr = NULL);
  void loadDataHeader(FILE* fileptr = NULL);
  void printData(unsigned int n = 10);
  void printSummary();
  void saveData(FILE* fileptr);
  void dumpData(FILE* fileptr, std::string format);
  void dumpLibsvm(FILE* fileptr);
  void dumpCSV(FILE* fileptr);
  void loadRankQuery();
  bool doesFileExist(std::string path);

	void constructAuxData();

 private:
  void adaptiveQuantization();
  unsigned short discretize(std::vector<double>& bin_starts, double value);
  void featureCleanUp();
  void loadLibsvmFormat(std::string path);
  void loadMatrixFormat(std::string path);
  void loadMemoryColumnMajorMatrix(double* Y_matrix, double* X_matrix, int nrow,
                                   int ncol);
  void loadMemoryKeyValueFormat(const double* Y_matrix,
          const std::vector<std::vector<std::pair<int,double>>>& X_kv,int n_row, int n_col);
  void normalizeLabels();
  void restoreDenseFeatures();
  bool testDataIsMatrix(std::string path);

  std::vector<std::string> split(const std::string& s, char delimiter = ',');
  std::vector<double> kmeans_value(std::vector<double>& value, int k);
  std::vector<double> find_bin_ckm(std::vector<double>& fv, size_t n_distinct,
                                   int max_n_bins);
  std::vector<double> find_bin_ckm_stratify(
      std::vector<std::vector<double>>& fv_strata, int max_n_bins,
      size_t n_distinct);
  std::vector<double> find_bin_fixed_size_binary_search(std::vector<double>& fv,
                                                        size_t n_distinct,
                                                        int max_n_bins);
  std::vector<double> find_bin_fixed_size_binary_search_stratify(
      std::vector<std::vector<double>>& fv_strata, int max_n_bins,
      size_t n_distinct);
};

}  // namespace ABCBoost

#endif  // ABCBOOST_DATA_H
