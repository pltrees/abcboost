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

#ifndef ABCBOOST_TREE_H
#define ABCBOOST_TREE_H

#include <utility>  // std::pair
#include <vector>

#include "config.h"
#include "data.h"
#include "utils.h"


namespace ABCBoost {
struct HistBin{
  int count;
  hist_t sum;
  hist_t weight;
  HistBin(){
    count = 0;
    sum = weight = 0;
  }
  HistBin(int c,hist_t s,hist_t w) : count(c),sum(s),weight(w){}
};

class Tree {
 public:
  struct SplitInfo {
    uint split_fi = 0;
    double gain = -1;
    int split_v = -1;
  };

  class TreeNode {
   public:
    bool is_leaf;
    short idx, left, right, parent;
    int start, end;
    // below are prediction related
    uint split_fi;
    double gain, predict_v;
    int split_v;

    TreeNode();

    bool operator<(const TreeNode &n) const { return gain < n.gain; }
  };

  // bin_counts stores the summed weights within a bin
  // bin_sums stores the summed residuals within a bin
  std::vector<std::vector<std::vector<HistBin>>> *hist;
  std::vector<std::vector<uint>> *l_buffer, *r_buffer;
  std::vector<double> *feature_importance;
  double *hessian, *residual;

  // ids stores the instance indices for each node
  std::vector<uint> ids;
  uint* ids_tmp;
  double* H_tmp;
  double* R_tmp;
  std::vector<uint> *fids;
  std::vector<int> dense_fids;
  std::vector<int> sparse_fids;

  // below are key properties of tree (saved after training)
  std::vector<short> leaf_ids;
  std::vector<TreeNode> nodes;
  bool is_weighted;
  int n_leaves, n_threads;

  Config *config;
  Data *data;

  std::vector<bool> in_leaf;

  Tree(Data *data, Config *config);
  ~Tree();

  virtual void binSort(int x, int sib);

  void buildTree(std::vector<uint> *ids, std::vector<uint> *fids);

  void updateFeatureImportance(int iter);

  std::pair<double, double> featureGain(int x, uint fid) const;

  void freeMemory();

  void init(std::vector<std::vector<uint>> *l_buffer,
            std::vector<std::vector<uint>> *r_buffer);

  void init(
      std::vector<std::vector<std::vector<HistBin>>>
          *hist,
      std::vector<std::vector<uint>> *l_buffer,
      std::vector<std::vector<uint>> *r_buffer,
      std::vector<double> *feature_importance, double *hessian, double *residual,
                uint* ids_tmp,
                double* H_tmp,
                double* R_tmp);

  void populateTree(const std::string line);
  void populateTree(FILE *fileptr);

  double predict(std::vector<ushort> instance);

  std::vector<double> predictAll(Data *data);

  void regress();

  void saveTree(FILE *fileptr);

  void split(int x, int l);

  void trySplit(int x, int sib);

  inline void alignHessianResidual(const uint start, const uint end);
  inline void initUnobserved(const uint start,const uint end,double& r_unobserved, double& h_unobserved);

  template<bool val>
  inline void setInLeaf(uint start,uint end){
  // Note: This is a hacky parallel for. Since in_leaf is a specialized vector<bool>,
  // parallel setting values whose indices are close can cause data race.
  // Here we force a chunk size to be large enough (4096) to dodge the data race.
  // We believe 4096 is large enough for most CPUs in 2022.
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 4096),
        config->use_omp == true && (end - start > 4096),
        for (int i = start; i < end; ++i) {
          in_leaf[ids[i]] = val;
        }
      )
  }
};

}  // namespace ABCBoost

#endif  // ABCBOOST_TREE_H
