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

#ifndef ABCBOOST_MODEL_GPU_H
#define ABCBOOST_MODEL_GPU_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "config.h"
#include "data.h"
#include "tree.h"
#include "model.h"

namespace ABCBoost {

class GradientBoostingGPU : public GradientBoosting{
 public:
  double* d_Y; 
  unsigned int* d_Xi;    // store the instance ids
  unsigned int** d_Xi_offset;
  unsigned short* d_Xv;  // store the corresponding value
  unsigned short** d_Xv_offset;
  bool* d_in_leaf;
  int* d_feature_id2idx;
  unsigned short* d_unobserved_fv;

  double* d_F;
  double* d_hessians;
  double* d_residuals;
  Tree::TreeNode* d_nodes;
  unsigned int* d_ids;
  int* d_fids;
  short* d_dense_f;

  int* d_hist_count;
  double* d_hist_sum;
  double* d_hist_weight;

  void* d_res;//32-byte temporary storage for storing results
  unsigned int* d_partition_id_out;
  unsigned int* d_temp_storage;
  unsigned int* d_prefix_sum;
  unsigned int* d_flags;
  double* d_max_gain;
  int* d_best_split_v;
  double* d_sum_numerator;
  double* d_sum_denominator;

  int* d_hist_count_batch;
  double* d_hist_sum_batch;
  double* d_hist_weight_batch;
  int* d_n_bins_per_f;
  
  int* d_cache_hist_count;
  double* d_cache_hist_sum;
  double* d_cache_hist_weight;
  int* d_accum_bins_start;
  int* d_total_bins;
  
  double* d_block_max_gain;
  int* d_block_split_fi;
  int* d_block_split_v;

  void* p_g_allocator;

  GradientBoostingGPU(Data* data, Config* config) : GradientBoosting(data,config){}
  void buildTree(int n,int k);
  void trySplitBatch(int m,int k,int x);
  void trySplitSub(int m,int k,int x,int parent,int sibling);
  void trySplit(int m,int k,int x);

  int computeErr();
  void print_train_message(int iter,double loss,double iter_time);
  
  virtual void init();
};

class MartGPU : public GradientBoostingGPU{
 public:
  MartGPU(Data* data, Config* config);
  void test();
  void train();

  
  void buildTrees(int n,int K);
  void computeLoss();

  void copyBackTrees(int M);
  void copyTreesToGPU(int M);

 private:
  void computeHessianResidual();
};

class ABCMartGPU : public GradientBoostingGPU{
 public:
  ABCMartGPU(Data* data, Config* config);
  void test();
  void train();

  double* d_class_loss;

  void buildTrees(int n,int K);
  void init();
  void computeLoss(int m);

  void copyBackTrees(int M);
  void copyTreesToGPU(int M);

  void updateBaseFValues(int b);
  void initBaseClass();

  void saveModel(int iter);
  int loadModel();

  void exhaustiveTrain(int n_skip);

 private:
  void computeHessianResidual();
  void computeHessianResidual(int m);
  void computeHessianResidual(int m,double* d_prob);
  void normalizeF(int m,int K,int n);
  std::vector<int> base_classes;
  std::vector<double> class_losses;
};

class RegressionGPU : public GradientBoostingGPU{
 public:
  RegressionGPU(Data* data, Config* config);
  void test();
  void train();

  void init();
  void computeLoss();

  void copyBackTrees(int M);
  void copyTreesToGPU(int M);
  void print_train_message(int iter,double loss,double iter_time);

 private:
  void computeHessianResidual();
  void computeHessianResidual(double* d_prob);
};

}  // namespace ABCBoost

#endif  // ABCBOOST_MODEL_H

