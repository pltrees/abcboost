#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "data.h"
#include "model_gpu.h"
#include "tree.h"
#include "utils.h"

#include <curand.h>

#include <cub/cub.cuh> 
#include <cub/util_allocator.cuh>

#ifdef OMP
#include "omp.h"
#else
#include "dummy_omp.h"
#endif

#define nB (256*8)  // nB >= 256 * 8
#define nT (16*32)  // data_max_n_bins <= nT <= 64 * 32
#define NUM_BLOCK 32
#define NUM_THREAD 256
#define FULL_MASK 0xffffffff

#define BATCH_NUM_BLOCK 512
#define BATCH_NUM_THREAD 128

#define INIT_NUM_THREAD 1024

namespace ABCBoost {

template<class Key,class Value>
struct ArgMaxPair{
  Key key;
  Value value;

  __host__ __device__ __forceinline__
  ArgMaxPair(){}

  __host__ __device__ __forceinline__
  ArgMaxPair(Key key,Value value) : key(key), value(value){}

  __host__ __device__ __forceinline__
  bool operator <(ArgMaxPair p) const{
    return key < p.key;
  }

  __host__ __device__ __forceinline__
  bool operator >(ArgMaxPair p) const{
    return key > p.key;
  }
};


//---------- CUDA kernels begin ----------//
__global__
void initNodeKernel(Tree::TreeNode* d_nodes,int n){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for(int i = gid;i < n;i += step){
    d_nodes[i].is_leaf = true;
    d_nodes[i].idx = -1;
    d_nodes[i].left = -1;
    d_nodes[i].right = -1;
    d_nodes[i].parent = -1;
    d_nodes[i].gain = -1;
    d_nodes[i].predict_v = -1;
  }
}

__global__
void fillKernel(double* d_F,long long len,double val){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for(long long i = gid;i < len;i += step)
    d_F[i] = val;
}

__global__
void fillIntKernel(int* d,long long len,int val){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for(long long i = gid;i < len;i += step)
    d[i] = val;
}

template<class T>
__global__
void sequenceKernel(T* d_ids,unsigned int len){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for(long long i = gid;i < len;i += step)
    d_ids[i] = i;
}

__global__
void printKernel(double* d_ptr,int cnt = 10){
  for(int i = 0;i < cnt;++i)
    printf("%d: %f\n",i,d_ptr[i]);
}

__global__
void printKernel(float* d_ptr,int cnt = 10){
  for(int i = 0;i < cnt;++i)
    printf("%d: %f\n",i,d_ptr[i]);
}

__global__
void printKernel(int* d_ptr,int cnt = 10){
  for(int i = 0;i < cnt;++i)
    printf("%d: %d\n",i,d_ptr[i]);
}

__global__
void printKernel(unsigned short* d_ptr,int cnt = 10){
  for(int i = 0;i < cnt;++i)
    printf("%d: %u\n",i,d_ptr[i]);
}

__global__
void printKernel(unsigned int* d_ptr,int cnt = 10){
  for(int i = 0;i < cnt;++i)
    printf("%d: %u\n",i,d_ptr[i]);
}

__global__
void updateF1Kernel(double* d_F,int n){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for(int i = gid;i < n;i += step)
    d_F[n + i] = -d_F[i];
}

__global__
void computeHessianResidualStaticMallocKernel(double* d_Y,double* d_F,double * d_residuals,double* d_hessians,unsigned int n_data,int n_classes,double* d_prob){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  double* prob = d_prob + gid * n_classes;
  for(unsigned int i = gid;i < n_data;i += step){
    int label = d_Y[i];
    for(int k = 0;k < n_classes;++k)
      prob[k] = d_F[k * n_data + i];
      
      double max = prob[0], normalization = 0;
      
      // find max value
      for (int j = 1; j < n_classes; ++j) {
        if(max < prob[j])
          max = prob[j];
      }

      for (int j = 0; j < n_classes; ++j) {
        double tmp = prob[j] - max;
        if (tmp > 700) tmp = 700;
        prob[j] = exp(tmp);
        normalization += prob[j];
      }

      // normalize
      for (int j = 0; j < n_classes; ++j) {
        prob[j] /= normalization;
      }
      for (int k = 0; k < n_classes; ++k) {
        double p_ik = prob[k];
        d_residuals[k * n_data + i] = (k == label) ? (1 - p_ik) : -p_ik;
        d_hessians[k * n_data + i] = p_ik * (1 - p_ik);
      }
  }
}
__global__
void computeHessianResidualKernel(double* d_Y,double* d_F,double * d_residuals,double* d_hessians,unsigned int n_data,int n_classes){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  double* prob = new double[n_classes]; 
  for(unsigned int i = gid;i < n_data;i += step){
    int label = d_Y[i];
    for(int k = 0;k < n_classes;++k)
      prob[k] = d_F[k * n_data + i];
      
      double max = prob[0], normalization = 0;
      
      // find max value
      for (int j = 1; j < n_classes; ++j) {
        if(max < prob[j])
          max = prob[j];
      }

      for (int j = 0; j < n_classes; ++j) {
        double tmp = prob[j] - max;
        if (tmp > 700) tmp = 700;
        prob[j] = exp(tmp);
        normalization += prob[j];
      }

      // normalize
      for (int j = 0; j < n_classes; ++j) {
        prob[j] /= normalization;
      }
      for (int k = 0; k < n_classes; ++k) {
        double p_ik = prob[k];
        d_residuals[k * n_data + i] = (k == label) ? (1 - p_ik) : -p_ik;
        d_hessians[k * n_data + i] = p_ik * (1 - p_ik);
      }
  }
  delete[] prob;
}

__global__
void computeRegressionHessianResidualKernel(double* d_Y,double* d_F,double * d_residuals,double* d_hessians,unsigned int n_data,const double p, const bool huber_loss, const double delta){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  if(huber_loss){
    const double delta_p_1 = pow(delta,p - 1);
    for(unsigned int i = gid;i < n_data;i += step){
      double label = d_Y[i];
      double pred = d_F[i];
      const double diff = fabs(pred - label);
      if(diff <= delta){
        d_residuals[i] = -p * pow(diff,p - 1);
        d_hessians[i] = 1;
      }else{
        d_residuals[i] = (pred - label > 0 ? -1.0 : 1.0) * delta_p_1;
        d_hessians[i] = 1;
      }
    }
  }else if(p == 2.0){
    for(unsigned int i = gid;i < n_data;i += step){
      double label = d_Y[i];
      double pred = d_F[i];
      int sign = (pred - label) > 0 ? -1 : 1;
      const double diff = fabs(pred - label);
      d_residuals[i] = 2.0 * diff * sign;
      d_hessians[i] = 2.0;
    }
  }else{
    for(unsigned int i = gid;i < n_data;i += step){
      double label = d_Y[i];
      double pred = d_F[i];
      int sign = (pred - label) > 0 ? -1 : 1;
      const double diff = fabs(pred - label);
      d_residuals[i] = p * pow(diff,p - 1) * sign;
      d_hessians[i] = (p <= 2) ? p : p * (p - 1) * pow(diff,p - 2);
    }
  }
}

__global__
void computeHessianResidualABCStaticMallocKernel(double* d_Y,double* d_F,double * d_residuals,double* d_hessians,unsigned int n_data,int n_classes,int b,double* d_prob){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  double* prob = d_prob + gid * n_classes;
  for(unsigned int i = gid;i < n_data;i += step){
    int label = d_Y[i];
    for(int k = 0;k < n_classes;++k)
      prob[k] = d_F[k * n_data + i];
      
      double max = prob[0], normalization = 0;
      
      // find max value
      for (int j = 1; j < n_classes; ++j) {
        if(max < prob[j])
          max = prob[j];
      }

      for (int j = 0; j < n_classes; ++j) {
        double tmp = prob[j] - max;
        if (tmp > 700) tmp = 700;
        prob[j] = exp(tmp);
        normalization += prob[j];
      }

      // normalize
      for (int j = 0; j < n_classes; ++j) {
        prob[j] /= normalization;
      }
      for (int k = 0; k < n_classes; ++k) {
        double p_ik = prob[k];
        double p_ib = prob[b];
        int r_ik = (int)(k == label);  // indicators used in calculations
        int r_ib = (int)(b == label);
        d_residuals[k * n_data + i] = p_ib - r_ib + r_ik - p_ik;
        d_hessians[k * n_data + i] = p_ik * (1 - p_ik) + p_ib * (1 - p_ib) + 2 * p_ib * p_ik;
      }
  }
}
__global__
void computeHessianResidualABCKernel(double* d_Y,double* d_F,double * d_residuals,double* d_hessians,unsigned int n_data,int n_classes,int b){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  double* prob = new double[n_classes];
  for(unsigned int i = gid;i < n_data;i += step){
    int label = d_Y[i];
    for(int k = 0;k < n_classes;++k)
      prob[k] = d_F[k * n_data + i];
      
      double max = prob[0], normalization = 0;
      
      // find max value
      for (int j = 1; j < n_classes; ++j) {
        if(max < prob[j])
          max = prob[j];
      }

      for (int j = 0; j < n_classes; ++j) {
        double tmp = prob[j] - max;
        if (tmp > 700) tmp = 700;
        prob[j] = exp(tmp);
        normalization += prob[j];
      }

      // normalize
      for (int j = 0; j < n_classes; ++j) {
        prob[j] /= normalization;
      }
      for (int k = 0; k < n_classes; ++k) {
        double p_ik = prob[k];
        double p_ib = prob[b];
        int r_ik = (int)(k == label);  // indicators used in calculations
        int r_ib = (int)(b == label);
        d_residuals[k * n_data + i] = p_ib - r_ib + r_ik - p_ik;
        d_hessians[k * n_data + i] = p_ik * (1 - p_ik) + p_ib * (1 - p_ib) + 2 * p_ib * p_ik;
      }
  }
  delete[] prob;
}

__global__
void buildTreeStartKernel(Tree::TreeNode* nodes,int tree_idx,int id_size){
  nodes[tree_idx].idx = 0;
  nodes[tree_idx].start = 0;
  nodes[tree_idx].end = id_size;
}

__global__
void trySplitBinSortKernel(Tree::TreeNode* d_nodes,unsigned int* d_ids,unsigned short* d_fv,int x,double* d_R,double* d_H,int* d_hist_count,double* d_hist_sum,double* d_hist_weight){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  auto start = d_nodes[x].start;
  auto end = d_nodes[x].end;
  for (uint i = start + gid; i < end; i += step) {
    uint id = d_ids[i];
    auto bin_id = d_fv[id];
    atomicAdd(&d_hist_count[bin_id],1);
    atomicAdd(&d_hist_sum[bin_id],d_R[id]);
    atomicAdd(&d_hist_weight[bin_id],d_H[id]);
  }
}


__global__
void trySplitFeatureGainKernel(Tree::TreeNode* d_nodes,int fid,int bin_size,int tree_min_node_size,int* d_hist_count,double* d_hist_sum,double* d_hist_weight){
  hist_t total_s = .0, total_w = .0;
  for (int i = 0; i < bin_size; ++i) {
    total_s += d_hist_sum[i];
    total_w += d_hist_weight[i];
  }

  int l_c = 0, r_c = 0;
  hist_t l_w = 0, l_s = 0;
  int st = 0, ed = ((int)bin_size) - 1;
  while (
      st <
      bin_size) {  // st = min_i (\sum_{k <= i} counts[i]) >= min_node_size
    l_c += d_hist_count[st];
    l_s += d_hist_sum[st];
    l_w += d_hist_weight[st];
    if (l_c >= tree_min_node_size) break;
    ++st;
  }

  if (st == bin_size) {
    return;
  }

  do {  // ed = max_i (\sum_{k > i} counts[i]) >= min_node_size
    r_c += d_hist_count[ed];
    ed--;
  } while (ed >= 0 && r_c < tree_min_node_size);

  if (st > ed) {
    //printf("return because st (%d) > ed (%d)\n",st,ed);
    return;
  }

  hist_t r_w = 0, r_s = 0;
  double max_gain = -1;
  int best_split_v = -1;
  for (int i = st; i <= ed; ++i) {
    if (d_hist_count[i] == 0) {
      if (i + 1 < bin_size) {
        l_w += d_hist_weight[i + 1];
        l_s += d_hist_sum[i + 1];
      }
      continue;
    }
    r_w = total_w - l_w;
    r_s = total_s - l_s;

    double gain = l_s / l_w * l_s + r_s / r_w * r_s;
    if (gain > max_gain ) {
      max_gain = gain;
      int offset = 1;
      while (i + offset < bin_size && d_hist_count[i + offset] == 0)
        offset++;
      best_split_v = i + offset / 2;
    }
    if (i + 1 < bin_size) {
      l_w += d_hist_weight[i + 1];
      l_s += d_hist_sum[i + 1];
    }
  }

  max_gain -= total_s / total_w * total_s;
  if(d_nodes->gain < max_gain){
    d_nodes->gain = max_gain;
    d_nodes->split_fi = fid;
    d_nodes->split_v = best_split_v;
//    printf("fid %d updated %f %d %d\n",fid,d_nodes->gain,d_nodes->split_fi,d_nodes->split_v);
  }else{
//    printf("fid %d not updated %f %d\n",fid,max_gain,best_split_v);
  }
  return;
}

__global__
void featureGainOnPrefixSumKernel(int* d_hist_count,double* d_hist_sum,double* d_hist_weight,int bin_size,double* d_max_gain,int* d_best_split_v,int tree_min_node_size){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  if(bin_size <= 0){
    printf("Error: found 0 size bin in featureGainOnPrefixSumKernel\n");
    return;
  }
  int total_c = d_hist_count[bin_size - 1];
  double total_s = d_hist_sum[bin_size - 1];
  double total_w = d_hist_weight[bin_size - 1];
  double max_gain = -1;
  int best_split_v = -1;
  for (int i = gid; i < bin_size; i += step) {
    const int l_c = d_hist_count[i];
    if(l_c < tree_min_node_size)
      continue;
    if(i > 0 && d_hist_count[i] == d_hist_count[i - 1])
      continue;
    if(total_c - l_c < tree_min_node_size)
      break;
  
    const double l_s = d_hist_sum[i];
    const double l_w = d_hist_weight[i];
    const double r_s = total_s - l_s;
    const double r_w = total_w - l_w;
    double gain = l_s / l_w * l_s + r_s / r_w * r_s;
    if (gain > max_gain ) {
      max_gain = gain;
      int offset = 1;
      while (i + offset < bin_size && d_hist_count[i + offset] == d_hist_count[i + offset - 1])
        offset++;
      best_split_v = i + offset / 2;
    }
  }
  if(max_gain >= 0)
    max_gain -= total_s / total_w * total_s;
  else{
  }
  d_max_gain[gid] = max_gain;
  d_best_split_v[gid] = best_split_v;
}

__global__
void trySplitZeroBinsKernel(int* d_hist_count,double* d_hist_sum,double* d_hist_weight,int bin_size){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for (int i = gid; i < bin_size; i += step) {
    d_hist_count[i] = 0;
    d_hist_sum[i] = 0;
    d_hist_weight[i] = 0;
  }
}

__global__
void trySplitBatchInitSparseKernel(Tree::TreeNode* d_nodes,unsigned int* d_ids,int x,double* d_R,double* d_H,double* d_r_unobserved,double* d_h_unobserved,bool* d_in_leaf){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;

  auto start = d_nodes[x].start;
  auto end = d_nodes[x].end;

  __shared__ double r_unobserved;
  __shared__ double h_unobserved;
  if(tid == 0){
    r_unobserved = 0;
    h_unobserved = 0;
  }
  __syncthreads();

  double local_r_unobserved = 0; 
  double local_h_unobserved = 0;
  for (int i = start + gid; i < end; i += step) {
    local_r_unobserved += d_R[d_ids[i]];
    local_h_unobserved += d_H[d_ids[i]];
  }
  atomicAdd(&r_unobserved,local_r_unobserved);
  atomicAdd(&h_unobserved,local_h_unobserved);

  __syncthreads();
  if(tid == 0){
    atomicAdd(d_r_unobserved,r_unobserved);
    atomicAdd(d_h_unobserved,h_unobserved);
  }
      
  for (int i = start + gid; i < end;i += step) {
    d_in_leaf[d_ids[i]] = true;
  }
}


__global__
void resetInLeafKernel(Tree::TreeNode* d_nodes,unsigned int* d_ids,int* px,bool* d_in_leaf){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  int x = *px;
  auto start = d_nodes[x].start;
  auto end = d_nodes[x].end;
      
  for (int i = start + gid; i < end;i += step) {
    d_in_leaf[d_ids[i]] = false;
  }
}

__global__
void resetFeatureIdxPxKernel(Tree::TreeNode* d_nodes,unsigned int* d_ids,int* px,bool* d_in_leaf,unsigned int** d_Xi_offset,int* d_feature_id2idx){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  int x = *px;
  if(x == -1){
    return;
  }
  auto fid = d_nodes[x].split_fi;  

  unsigned int* d_fi = d_Xi_offset[fid];
  int len = d_Xi_offset[fid + 1] - d_Xi_offset[fid];

  for (int i = gid; i < len;i += step) {
    d_feature_id2idx[d_fi[i]] = -1;
  }
}
__global__
void setFeatureIdxPxKernel(Tree::TreeNode* d_nodes,unsigned int* d_ids,int* px,bool* d_in_leaf,unsigned int** d_Xi_offset,int* d_feature_id2idx){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  int x = *px;
  if(x == -1 && gid == 0){
    printf("[INFO] cannot split further.\n");
  }
  if(x == -1){
    return;
  }
  auto fid = d_nodes[x].split_fi;  

  unsigned int* d_fi = d_Xi_offset[fid];
  int len = d_Xi_offset[fid + 1] - d_Xi_offset[fid];

  for (int i = gid; i < len;i += step) {
    d_feature_id2idx[d_fi[i]] = i;
  }
}

__global__
void resetInLeafKernel(Tree::TreeNode* d_nodes,unsigned int* d_ids,int x,bool* d_in_leaf){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  auto start = d_nodes[x].start;
  auto end = d_nodes[x].end;
      
  for (int i = start + gid; i < end;i += step) {
    d_in_leaf[d_ids[i]] = false;
  }
}

__global__
void trySplitBatchSubKernel(Tree::TreeNode* d_nodes,unsigned int* d_ids,unsigned short** d_Xv_offset,int x,double* d_R,double* d_H,int* d_hist_count_batch,double* d_hist_sum_batch,double* d_hist_weight_batch,int* d_fids,int n_sample_feats,int n_feats,int* d_n_bins_per_f,int max_bin,int tree_min_node_size,unsigned int n_data,double* d_block_max_gain,int* d_block_split_fi,int* d_block_split_v,double* d_r_unobserved,double* d_h_unobserved,bool* d_in_leaf,short* d_dense_f,unsigned int** d_Xi_offset,unsigned short* d_unobserved_fv,int* d_cache_hist_count,double* d_cache_hist_sum,double* d_cache_hist_weight,int parent,int sibling,int* d_accum_bins_start, int* d_total_bins){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  auto start = d_nodes[x].start;
  auto end = d_nodes[x].end;

  double block_max_gain = -1;
  int block_split_fi = -1;
  int block_split_v = -1;


  double r_unobserved = *d_r_unobserved;
  double h_unobserved = *d_h_unobserved;
  int c_unobserved = end - start;


  for(int jj = bid;jj < n_sample_feats;jj += BATCH_NUM_BLOCK){
    int j = d_fids[jj];
    int bin_size = d_n_bins_per_f[j];
    if(bin_size == 0){
  //    printf("skiping %d becuase bin_size is 0\n",j);
      continue;
    }
    int item_per_thread = bin_size / BATCH_NUM_THREAD;
    if(bin_size % BATCH_NUM_THREAD != 0)
      item_per_thread += 1;

    int* d_hist_count = d_hist_count_batch + bid * max_bin;
    double* d_hist_sum = d_hist_sum_batch + bid * max_bin;
    double* d_hist_weight = d_hist_weight_batch + bid * max_bin;
    unsigned short* d_fv = d_Xv_offset[j];
    
    int total_bins = *d_total_bins;
    // get prefix sum by subtraction
    for (int i = tid; i < bin_size; i += BATCH_NUM_THREAD) {
      d_hist_count[i] = d_cache_hist_count[parent * total_bins + d_accum_bins_start[jj] + i] - d_cache_hist_count[sibling * total_bins + d_accum_bins_start[jj] + i];
      d_hist_sum[i] = d_cache_hist_sum[parent * total_bins + d_accum_bins_start[jj] + i] - d_cache_hist_sum[sibling * total_bins + d_accum_bins_start[jj] + i];
      d_hist_weight[i] = d_cache_hist_weight[parent * total_bins + d_accum_bins_start[jj] + i] - d_cache_hist_weight[sibling * total_bins + d_accum_bins_start[jj] + i];
    }
    __syncthreads();


    //-----copy prefix hist for histogram substraction-----//
    for(int i = tid;i < bin_size;i += BATCH_NUM_THREAD){
      d_cache_hist_count[x * total_bins + d_accum_bins_start[jj] + i] = d_hist_count[i];
      d_cache_hist_sum[x * total_bins + d_accum_bins_start[jj] + i] = d_hist_sum[i];
      d_cache_hist_weight[x * total_bins + d_accum_bins_start[jj] + i] = d_hist_weight[i];
    }
    //----------//


    __syncthreads();

    // feature gain on prefix sum
    int total_c = d_hist_count[bin_size - 1];
    double total_s = d_hist_sum[bin_size - 1];
    double total_w = d_hist_weight[bin_size - 1];
    double max_gain = -1;
    int best_split_v = -1;
    for (int i = tid; i < bin_size; i += BATCH_NUM_THREAD) {
      const int l_c = d_hist_count[i];
      if(l_c < tree_min_node_size)
        continue;
      if(i > 0 && d_hist_count[i] == d_hist_count[i - 1])
        continue;
      if(total_c - l_c < tree_min_node_size)
        break;
    
      const double l_s = d_hist_sum[i];
      const double l_w = d_hist_weight[i];
      const double r_s = total_s - l_s;
      const double r_w = total_w - l_w;
      double gain = l_s / l_w * l_s + r_s / r_w * r_s;
      if (gain > max_gain ) {
        max_gain = gain;
        int offset = 1;
        while (i + offset < bin_size && d_hist_count[i + offset] == d_hist_count[i + offset - 1])
          offset++;
        best_split_v = i + offset / 2;
      }
    }
    if(max_gain >= 0)
      max_gain -= total_s / total_w * total_s;
    if(block_max_gain < max_gain){
      block_max_gain = max_gain;
      block_split_fi = j;
      block_split_v = best_split_v;
    }
  }
  d_block_max_gain[gid] = block_max_gain;
  d_block_split_fi[gid] = block_split_fi;
  d_block_split_v[gid] = block_split_v;
}

template<bool use_logit>
__global__
void trySplitBatchKernel(Tree::TreeNode* d_nodes,unsigned int* d_ids,unsigned short** d_Xv_offset,int x,double* d_R,double* d_H,int* d_hist_count_batch,double* d_hist_sum_batch,double* d_hist_weight_batch,int* d_fids,int n_sample_feats,int n_feats,int* d_n_bins_per_f,int max_bin,int tree_min_node_size,unsigned int n_data,double* d_block_max_gain,int* d_block_split_fi,int* d_block_split_v,double* d_r_unobserved,double* d_h_unobserved,bool* d_in_leaf,short* d_dense_f,unsigned int** d_Xi_offset,unsigned short* d_unobserved_fv,int* d_cache_hist_count,double* d_cache_hist_sum,double* d_cache_hist_weight,int* d_accum_bins_start, int* d_total_bins){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  auto start = d_nodes[x].start;
  auto end = d_nodes[x].end;

  double block_max_gain = -1;
  int block_split_fi = -1;
  int block_split_v = -1;


  double r_unobserved = *d_r_unobserved;
  double h_unobserved = *d_h_unobserved;
  int c_unobserved = end - start;


  for(int jj = bid;jj < n_sample_feats;jj += BATCH_NUM_BLOCK){
    int j = d_fids[jj];
    int bin_size = d_n_bins_per_f[j];
    if(bin_size == 0){
      continue;
    }
    int item_per_thread = bin_size / BATCH_NUM_THREAD;
    if(bin_size % BATCH_NUM_THREAD != 0)
      item_per_thread += 1;

    int* d_hist_count = d_hist_count_batch + bid * max_bin;
    double* d_hist_sum = d_hist_sum_batch + bid * max_bin;
    double* d_hist_weight = d_hist_weight_batch + bid * max_bin;
    unsigned short* d_fv = d_Xv_offset[j];
    

    // zero bins
    for (int i = tid; i < bin_size; i += BATCH_NUM_THREAD) {
      d_hist_count[i] = 0;
      d_hist_sum[i] = 0;
      d_hist_weight[i] = 0;
    }
    __syncthreads();

    // bin sort
    if(d_dense_f[j]){
      for (uint i = start + tid; i < end; i += BATCH_NUM_THREAD) {
        uint id = d_ids[i];
        auto bin_id = d_fv[id];
        atomicAdd(&d_hist_count[bin_id],1);
        atomicAdd(&d_hist_sum[bin_id],d_R[id]);
        atomicAdd(&d_hist_weight[bin_id],use_logit ? d_H[id] : 1);
      }
      __syncthreads();
    }else{  //sparse feature
      unsigned int* d_fi = d_Xi_offset[j];
      int len = d_Xi_offset[j + 1] - d_Xi_offset[j];
      ushort j_unobserved = d_unobserved_fv[j];
      if(tid == 0){
        d_hist_count[j_unobserved] = c_unobserved;
        d_hist_sum[j_unobserved] = r_unobserved;
        d_hist_weight[j_unobserved] = use_logit ? h_unobserved : c_unobserved;
      }
      __syncthreads();

      for(int i = tid;i < len;i += BATCH_NUM_THREAD){
        if(d_in_leaf[d_fi[i]] == true){
          auto bin_id = d_fv[i];
          auto id = d_fi[i];
          atomicAdd(&d_hist_count[bin_id],1);
          atomicAdd(&d_hist_sum[bin_id],d_R[id]);
          atomicAdd(&d_hist_weight[bin_id],use_logit ? d_H[id] : 1);

          atomicAdd(&d_hist_count[j_unobserved],-1);
          atomicAdd(&d_hist_sum[j_unobserved],-d_R[id]);
          atomicAdd(&d_hist_weight[j_unobserved],use_logit ? -d_H[id] : -1);
        }
      }
    }
    __syncthreads();
    
    // prefix sum 
    if(tid == 0){
      for(int i = 1;i < bin_size;++i)
        d_hist_count[i] += d_hist_count[i - 1];
      for(int i = 1;i < bin_size;++i)
        d_hist_sum[i] += d_hist_sum[i - 1];
      for(int i = 1;i < bin_size;++i)
        d_hist_weight[i] += d_hist_weight[i - 1];
    }
    __syncthreads();

    int total_bins = *d_total_bins;
    //-----copy prefix hist for histogram substraction-----//
    for(int i = tid;i < bin_size;i += BATCH_NUM_THREAD){
      d_cache_hist_count[x * total_bins + d_accum_bins_start[jj] + i] = d_hist_count[i];
      d_cache_hist_sum[x * total_bins + d_accum_bins_start[jj] + i] = d_hist_sum[i];
      d_cache_hist_weight[x * total_bins + d_accum_bins_start[jj] + i] = d_hist_weight[i];
    }
    //----------//


    __syncthreads();

    // feature gain on prefix sum
    int total_c = d_hist_count[bin_size - 1];
    double total_s = d_hist_sum[bin_size - 1];
    double total_w = d_hist_weight[bin_size - 1];
    double max_gain = -1;
    int best_split_v = -1;
    for (int i = tid; i < bin_size; i += BATCH_NUM_THREAD) {
      const int l_c = d_hist_count[i];
      if(l_c < tree_min_node_size)
        continue;
      if(i > 0 && d_hist_count[i] == d_hist_count[i - 1])
        continue;
      if(total_c - l_c < tree_min_node_size)
        break;
    
      const double l_s = d_hist_sum[i];
      const double l_w = d_hist_weight[i];
      const double r_s = total_s - l_s;
      const double r_w = total_w - l_w;
      double gain = l_s / l_w * l_s + r_s / r_w * r_s;
      if (gain > max_gain ) {
        max_gain = gain;
        int offset = 1;
        while (i + offset < bin_size && d_hist_count[i + offset] == d_hist_count[i + offset - 1])
          offset++;
        best_split_v = i + offset / 2;
      }
    }
    if(max_gain >= 0)
      max_gain -= total_s / total_w * total_s;
    if(block_max_gain < max_gain){
      block_max_gain = max_gain;
      block_split_fi = j;
      block_split_v = best_split_v;
    }
  }
  d_block_max_gain[gid] = block_max_gain;
  d_block_split_fi[gid] = block_split_fi;
  d_block_split_v[gid] = block_split_v;
}

__global__
void trySplitBatchFinalizeKernel(Tree::TreeNode* d_nodes,double* d_block_max_gain,int* d_block_split_fi,int* d_block_split_v){
  int tid = threadIdx.x;

  typedef cub::BlockReduce<ArgMaxPair<double,int>, NUM_THREAD> PairBlockReduce;
  __shared__ typename PairBlockReduce::TempStorage pair_temp_storage;
  double gain = -1;
  int id = -1;
  for(int i = tid;i < BATCH_NUM_BLOCK * BATCH_NUM_THREAD;i += NUM_THREAD){
    if(gain < d_block_max_gain[i]){
      gain = d_block_max_gain[i];
      id = i;
    }
  }

  ArgMaxPair<double,int> argmax(gain,id);
  __syncthreads();
  argmax = PairBlockReduce(pair_temp_storage).Reduce(argmax, cub::Max());
  __syncthreads();
  if(tid == 0){
    if(d_nodes->gain < argmax.key){
      d_nodes->gain = argmax.key;
      d_nodes->split_fi = d_block_split_fi[argmax.value];
      d_nodes->split_v = d_block_split_v[argmax.value];
    }
  }
  
}

__global__
void findSplitIdxKernel(Tree::TreeNode* d_nodes,int i,int* d_res){
  int idx = -1;
  double max_gain = -1;
  for(int j = 0;j < 2 * i + 1;++j){
    if(d_nodes[j].is_leaf && d_nodes[j].gain > max_gain){
      idx = j;
      max_gain = d_nodes[j].gain;
    }
  }
  *d_res = idx;
  d_res[1] = d_nodes[idx].start;
  d_res[2] = d_nodes[idx].end - d_res[1];
}

__global__
void splitFinalizeKernel(Tree::TreeNode* d_nodes,int l,int num_selected){
  d_nodes[l].end = d_nodes[l].start + num_selected;
  d_nodes[l + 1].start = d_nodes[l].start + num_selected;
}

__global__
void splitSetFlagKernel(Tree::TreeNode* d_nodes,int* px,int l,unsigned short** d_Xv_offset,unsigned int n_data,unsigned int* d_ids,unsigned int* d_temp_storage,unsigned int* d_flags,short* d_dense_f,unsigned short* d_unobserved_fv,int* d_feature_id2idx){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  
  const int x = *px;
  if(x == -1){
    return;
  }
  uint pstart = d_nodes[x].start;
  uint pend = d_nodes[x].end;
  int split_v = d_nodes[x].split_v;
  uint fid = d_nodes[x].split_fi;
  unsigned short* d_fv = d_Xv_offset[fid];

  if(gid == 0){
    d_nodes[x].is_leaf = false;
    d_nodes[x].left = l;
    d_nodes[x].right = l + 1;
    d_nodes[l].idx = l;
    d_nodes[l].parent = x;
    d_nodes[l + 1].idx = l + 1;
    d_nodes[l + 1].parent = x;
    d_nodes[l].start = pstart;
    d_nodes[l + 1].end = pend;
  }

  if(d_dense_f[fid]){
    for (int i = pstart + gid;i < pend;i += step) {
      uint id = d_ids[i];
      d_temp_storage[i] = d_ids[i];
      d_flags[i] = (d_fv[id] <= split_v) ? 1 : 0; 
    }
  }else{
    int unobserved_flag = ((d_unobserved_fv)[fid] <= split_v) ? 1 : 0;
    for (int i = pstart + gid;i < pend;i += step) {
      uint id = d_ids[i];
      d_temp_storage[i] = d_ids[i];
      int idx = d_feature_id2idx[id];
      if(idx == -1){
        d_flags[i] = unobserved_flag;
      }else{
        d_flags[i] = (d_fv[idx] <= split_v) ? 1 : 0; 
      }
    }
  }
}

__global__
void regressKernel(Tree::TreeNode* d_nodes,int i,int n_nodes,unsigned int* d_ids,double* d_R,double* d_H,double correction, double tree_clip_value, double tree_damping_factor,double* d_sum_numerator,double* d_sum_denominator){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;

  double numerator = 0.0, denominator = 0.0;
  if (d_nodes[i].idx >= 0 && d_nodes[i].is_leaf) {
    uint start = d_nodes[i].start, end = d_nodes[i].end;
    for (uint d = start + gid; d < end; d += step) {
      uint id = d_ids[d];
      numerator += d_R[id];
      denominator += d_H[id];
    }
  }
  d_sum_numerator[gid] = numerator;
  d_sum_denominator[gid] = denominator;
}

__global__
void regressFinalizeKernel(Tree::TreeNode* d_nodes,int i,double correction, double tree_clip_value, double tree_damping_factor,double* d_numerator,double* d_denominator){
  double numerator = *d_numerator;
  double denominator = *d_denominator;

  double upper = tree_clip_value, lower = -upper;

  if (d_nodes[i].idx >= 0 && d_nodes[i].is_leaf) {
    d_nodes[i].predict_v =
        min(max(correction * numerator /
                 (denominator + tree_damping_factor),
               lower),
           upper);
  }
}

__global__
void updateFKernel(Tree::TreeNode* d_nodes,int n_nodes,double* d_f,unsigned int* d_ids,double model_shrinkage){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for(int x = 0;x < n_nodes;++x){
    if(d_nodes[x].is_leaf){
      double update = model_shrinkage * d_nodes[x].predict_v;
      auto start = d_nodes[x].start;
      auto end = d_nodes[x].end;
      for(int i = start + gid;i < end;i += step){
        d_f[d_ids[i]] += update;
      }
    }
  }
}

__global__
void computeLossKernel(double* d_Y,unsigned int n_data,double* d_F,int n_classes,double* d_res){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  __shared__ double total_loss;
  if(tid == 0)
    total_loss = 0;
  __syncthreads();
  double loss = 0.0;
  for (int i = gid; i < n_data; i += step) {
    if (d_Y[i] >= 0) {
      double curr = d_F[int(d_Y[i]) * n_data + i];
      double denominator = 0;
      for (int k = 0; k < n_classes; ++k) {
        double tmp = d_F[k * n_data + i] - curr;
        if (tmp > 700) tmp = 700;
        denominator += exp(tmp);
      }
      // get loss for one example and add it to the total
      loss += log(denominator);
    }
  }
  atomicAdd(&total_loss,loss);
  __syncthreads();
  if(tid == 0){
    atomicAdd(d_res,total_loss);
  }
}

__global__
void computeLpLossKernel(double* d_Y,unsigned int n_data,double* d_F,const double p,double* d_res){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  __shared__ double total_loss;
  if(tid == 0)
    total_loss = 0;
  __syncthreads();
  double loss = 0.0;
  if(p == 1){
    for (int i = gid; i < n_data; i += step) {
      double diff = fabs(d_F[i] - d_Y[i]);
      loss += diff;
    }
  }else if(p == 2){
    for (int i = gid; i < n_data; i += step) {
      double diff = d_F[i] - d_Y[i];
      loss += diff * diff;
    }
  }else{
    for (int i = gid; i < n_data; i += step) {
      double diff = fabs(d_F[i] - d_Y[i]);
      loss += pow(diff,p);
    }
  }
  atomicAdd(&total_loss,loss);
  __syncthreads();
  if(tid == 0){
    atomicAdd(d_res,total_loss);
  }
}

__global__
void computeHuberLossKernel(double* d_Y,unsigned int n_data,double* d_F,const double p,const double delta,double* d_res){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  __shared__ double total_loss;
  if(tid == 0)
    total_loss = 0;
  __syncthreads();
  double loss = 0.0;
  const double delta_p_1 = pow(delta,p - 1);
  for (int i = gid; i < n_data; i += step) {
    double diff = fabs(d_F[i] - d_Y[i]);
    if(diff <= delta){
      loss += pow(diff,p);
    }else{
      loss += delta_p_1 * (2 * diff - delta);
    }
  }
  atomicAdd(&total_loss,loss);
  __syncthreads();
  if(tid == 0){
    atomicAdd(d_res,total_loss);
  }
}

__global__
void computeErrorKernel(double* d_Y,unsigned int n_data,double* d_F,int n_classes,int* d_err){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  __shared__ int total_err;
  if(tid == 0){
    total_err = 0;
  }
  __syncthreads();
  int local_err = 0;
  for (int i = gid; i < n_data; i += step) {
    if (d_Y[i] >= 0) {
      double max_f = d_F[i];
      int pred_k = 0;
      for (int k = 0; k < n_classes; ++k) {
        if(max_f < d_F[k * n_data + i]){
          max_f = d_F[k * n_data + i];
          pred_k = k;
        }
      }
      // get loss for one example and add it to the total
      if(pred_k != (int)d_Y[i])
        ++local_err;
    }
  }
  atomicAdd(&total_err,local_err);
  __syncthreads();
  if(tid == 0){
    atomicAdd(d_err,total_err);
  }
}

__global__
void computeLossAndClassLossAndErrorKernel(double* d_Y,unsigned int n_data,double* d_F,int n_classes,double* d_class_loss,double* d_res,int* d_err){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  __shared__ double total_loss;
  __shared__ int total_err;
  if(tid == 0){
    total_loss = 0;
    total_err = 0;
  }
  __syncthreads();
  double loss = 0.0;
  int local_err = 0;
  for (int i = gid; i < n_data; i += step) {
    if (d_Y[i] >= 0) {
      double curr = d_F[int(d_Y[i]) * n_data + i];
      double denominator = 0;
      double max_f = d_F[i];
      int pred_k = 0;
      for (int k = 0; k < n_classes; ++k) {
        if(max_f < d_F[k * n_data + i]){
          max_f = d_F[k * n_data + i];
          pred_k = k;
        }
        double tmp = d_F[k * n_data + i] - curr;
        if (tmp > 700) tmp = 700;
        denominator += exp(tmp);
      }
      // get loss for one example and add it to the total
      double loss_i = log(denominator);
      loss += loss_i;
      atomicAdd(&d_class_loss[(int)d_Y[i]],loss_i);
      if(pred_k != (int)d_Y[i])
        ++local_err;
    }
  }
  atomicAdd(&total_loss,loss);
  atomicAdd(&total_err,local_err);
  __syncthreads();
  if(tid == 0){
    atomicAdd(d_res,total_loss);
    atomicAdd(d_err,total_err);
  }
}
__global__
void computeLossAndClassLossKernel(double* d_Y,unsigned int n_data,double* d_F,int n_classes,double* d_class_loss,double* d_res){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  __shared__ double total_loss;
  if(tid == 0)
    total_loss = 0;
  __syncthreads();
  double loss = 0.0;
  for (int i = gid; i < n_data; i += step) {
    if (d_Y[i] >= 0) {
      double curr = d_F[int(d_Y[i]) * n_data + i];
      double denominator = 0;
      for (int k = 0; k < n_classes; ++k) {
        double tmp = d_F[k * n_data + i] - curr;
        if (tmp > 700) tmp = 700;
        denominator += exp(tmp);
      }
      // get loss for one example and add it to the total
      double loss_i = log(denominator);
      loss += loss_i;
      atomicAdd(&d_class_loss[(int)d_Y[i]],loss_i);
    }
  }
  atomicAdd(&total_loss,loss);
  __syncthreads();
  if(tid == 0){
    atomicAdd(d_res,total_loss);
  }
}
//---------- CUDA kernels end ----------//



void GradientBoostingGPU::init(){
  GradientBoosting::init();
  int device_count = 0;
  auto err = cudaGetDeviceCount(&device_count);
  if(err != cudaSuccess){
    printf("[ERROR] Failed to find GPUs. %s: %s\n",cudaGetErrorName(err),cudaGetErrorString(err));
    throw std::runtime_error("Failed to find GPUs");
  }
  cudaMalloc(&d_nodes,sizeof(Tree::TreeNode) * config->model_n_iterations * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
  initNodeKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes,config->model_n_iterations * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
  cudaMalloc(&d_hist_count,sizeof(int) * (config->data_max_n_bins));
  cudaMalloc(&d_hist_sum,sizeof(double) * (config->data_max_n_bins));
  cudaMalloc(&d_hist_weight,sizeof(double) * (config->data_max_n_bins));
  cudaMalloc(&d_res,32);
  cudaMalloc(&d_partition_id_out,sizeof(unsigned int) * data->n_data);
  cudaMalloc(&d_temp_storage,sizeof(unsigned int) * data->n_data);
  cudaMalloc(&d_prefix_sum,sizeof(unsigned int) * data->n_data);
  cudaMalloc(&d_flags,sizeof(unsigned int) * data->n_data);
  cudaMalloc(&d_max_gain,sizeof(double) * NUM_BLOCK * NUM_THREAD);
  cudaMalloc(&d_best_split_v,sizeof(int) * NUM_BLOCK * NUM_THREAD);
  cudaMalloc(&d_sum_numerator,sizeof(double) * NUM_BLOCK * NUM_THREAD);
  cudaMalloc(&d_sum_denominator,sizeof(double) * NUM_BLOCK * NUM_THREAD);
  
  cudaMalloc(&d_hist_count_batch,sizeof(int) * (config->data_max_n_bins) * BATCH_NUM_BLOCK);
  cudaMalloc(&d_hist_sum_batch,sizeof(double) * (config->data_max_n_bins) * BATCH_NUM_BLOCK);
  cudaMalloc(&d_hist_weight_batch,sizeof(double) * (config->data_max_n_bins) * BATCH_NUM_BLOCK);


  cudaMalloc(&d_n_bins_per_f,sizeof(int) * data->data_header.n_feats);
  cudaMemcpy(d_n_bins_per_f,data->data_header.n_bins_per_f.data(),sizeof(int) * data->data_header.n_feats,cudaMemcpyHostToDevice);

  cudaMalloc(&d_block_max_gain,sizeof(double) * BATCH_NUM_BLOCK * BATCH_NUM_THREAD);
  cudaMalloc(&d_block_split_fi,sizeof(int) * BATCH_NUM_BLOCK * BATCH_NUM_THREAD);
  cudaMalloc(&d_block_split_v,sizeof(int) * BATCH_NUM_BLOCK * BATCH_NUM_THREAD);

  cudaMalloc(&d_in_leaf,sizeof(bool) * data->n_data);
  cudaMalloc(&d_feature_id2idx,sizeof(int) * data->n_data);
  fillIntKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_feature_id2idx,data->n_data,-1);

  cudaMalloc(&d_unobserved_fv,sizeof(unsigned short) * data->data_header.unobserved_fv.size());
  cudaMemcpy(d_unobserved_fv,data->data_header.unobserved_fv.data(),sizeof(unsigned short) * data->data_header.unobserved_fv.size(),cudaMemcpyHostToDevice);


  cudaMalloc(&d_ids,sizeof(unsigned int) * data->n_data);
  cudaMalloc(&d_fids,sizeof(unsigned int) * data->data_header.n_feats);

  cudaMalloc(&d_F,sizeof(double) * data->data_header.n_classes * data->n_data);
  cudaMalloc(&d_Y,sizeof(double) * data->n_data);
  
  size_t total_Xv_size = 0;
  for(int j = 0;j < data->Xv.size();++j)
    total_Xv_size += data->Xv[j].size();

  size_t total_Xi_size = 0;
  for(int j = 0;j < data->Xv.size();++j)
    if(data->dense_f[j] == 0)
      total_Xi_size += data->Xi[j].size();

  cudaMalloc(&d_Xv,sizeof(ushort) * total_Xv_size);

  cudaMalloc(&d_residuals,sizeof(double) * data->data_header.n_classes * data->n_data);
  cudaMalloc(&d_hessians,sizeof(double) * data->data_header.n_classes * data->n_data);
  
  cudaMalloc(&d_dense_f,sizeof(short) * data->Xv.size());
  cudaMemcpy(d_dense_f,data->dense_f.data(),sizeof(short) * data->Xv.size(),cudaMemcpyHostToDevice);
  
  std::vector<unsigned int*> Xi_offset(data->Xv.size() + 1);
  cudaMalloc(&d_Xi,sizeof(unsigned int) * total_Xi_size);
  unsigned int* p_d_Xi = d_Xi;
  for(int j = 0;j < data->Xv.size();++j){
    if(data->dense_f[j]){
      Xi_offset[j] = p_d_Xi;
    }else{
      cudaMemcpy(p_d_Xi,data->Xi[j].data(),sizeof(unsigned int) * data->Xi[j].size(),cudaMemcpyHostToDevice);
      Xi_offset[j] = p_d_Xi;
      p_d_Xi += data->Xi[j].size();
    }
  }
  Xi_offset[data->Xv.size()] = p_d_Xi;
  cudaMalloc(&d_Xi_offset,sizeof(unsigned int*) * (data->Xv.size() + 1));
  cudaMemcpy(d_Xi_offset,Xi_offset.data(),sizeof(unsigned int*) * (data->Xv.size() + 1),cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_Y, data->Y.data(), sizeof(double) * data->Y.size(), cudaMemcpyHostToDevice);

  std::vector<unsigned short*> Xv_offset(data->Xv.size());
  unsigned short* p_d_Xv = d_Xv;
  for(int j = 0;j < data->Xv.size();++j){
    cudaMemcpy(p_d_Xv,data->Xv[j].data(),sizeof(ushort) * data->Xv[j].size(),cudaMemcpyHostToDevice);
    Xv_offset[j] = p_d_Xv;
    p_d_Xv += data->Xv[j].size();
  }
  cudaMalloc(&d_Xv_offset,sizeof(unsigned short*) * Xv_offset.size());
  cudaMemcpy(d_Xv_offset,Xv_offset.data(),sizeof(unsigned short*) * Xv_offset.size(),cudaMemcpyHostToDevice);
  fillKernel<<<nB,nT>>>(d_F,1LL * data->data_header.n_classes * data->n_data,0);


  std::vector<int> accum_bins_start;
  int sum = 0;
  for(int i = 0;i < data->data_header.n_bins_per_f.size();++i){
    accum_bins_start.push_back(sum);
    sum += data->data_header.n_bins_per_f[i];
  }
  cudaMalloc(&d_cache_hist_count,sizeof(int) * (2 * config->tree_max_n_leaves - 1) * sum);
  cudaMalloc(&d_cache_hist_sum,sizeof(double) * (2 * config->tree_max_n_leaves - 1) * sum);
  cudaMalloc(&d_cache_hist_weight,sizeof(double) * (2 * config->tree_max_n_leaves - 1) * sum);

  cudaMalloc(&d_accum_bins_start,sizeof(int) * accum_bins_start.size());
  cudaMemcpy(d_accum_bins_start,accum_bins_start.data(),sizeof(int) * accum_bins_start.size(),cudaMemcpyHostToDevice);
  
  cudaMalloc(&d_total_bins,sizeof(int));
  cudaMemcpy(d_total_bins,&sum,sizeof(int),cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  err = cudaPeekAtLastError();
  if(err != cudaSuccess){
    printf("[ERROR] Failed to allocate GPU memory. No sufficient memory? Try to use less iteration or terminal nodes. %s: %s\n",cudaGetErrorName(err),cudaGetErrorString(err));
    throw std::runtime_error("Failed to allocate GPU memory");
  }
}

//----------Mart begin----------//

MartGPU::MartGPU(Data *data, Config *config) : GradientBoostingGPU(data, config) {
  p_g_allocator = reinterpret_cast<void*>(new cub::CachingDeviceAllocator());
}

/**
 * Method to implement testing process for MART algorithm as described by
 * Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 */
void MartGPU::test() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  Utils::Timer t1;
  t1.restart();

  double best_accuracy = 0;
  int K = (data->data_header.n_classes == 2) ? 1 : data->data_header.n_classes;
  for (int m = 0; m < config->model_n_iterations; ++m) {
    for (int k = 0; k < K; ++k) {
      if (additive_trees[m][k] != NULL) {
        additive_trees[m][k]->init(nullptr, &buffer[0], &buffer[1], nullptr,
                                   nullptr, nullptr,
                                   nullptr,nullptr,nullptr);
        std::vector<double> updates = additive_trees[m][k]->predictAll(data);
        for (int i = 0; i < data->n_data; ++i) {
          F[k][i] += config->model_shrinkage * updates[i];
        }
      }
    }

    if (data->data_header.n_classes == 2) {
#pragma omp parallel for
      for (int i = 0; i < data->n_data; ++i) F[1][i] = -F[0][i];
    }

    double acc = getAccuracy();
    if (acc > best_accuracy) best_accuracy = acc;
    if ((m + 1) % config->model_eval_every == 0)
      printf("%d\tTime: %f | acc/best_acc: %f/%f | loss: %f \n", m + 1,
             t1.get_time_restart(), acc, best_accuracy, getLoss());
  }

  for(int k = 0;k < data->data_header.n_classes;++k)
    cudaMemcpy(d_F + k * data->n_data,F[k].data(),sizeof(double) * data->n_data,cudaMemcpyHostToDevice);
  copyTreesToGPU(config->model_n_iterations);
}

/**
 * Method to implement training process for MART algorithm as described
 * by Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 * @param[in] start: index to start training at.
 */



void MartGPU::copyTreesToGPU(int M){
  std::vector<Tree::TreeNode> nodes_buffer(M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
  
  int K = (data->data_header.n_classes == 2) ? 1 : data->data_header.n_classes;
  for(int m = 0;m < M;++m){
    for(int k = 0;k < K;++k){
      const int tree_idx = m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1) + k * (2 * config->tree_max_n_leaves - 1);
      memcpy(nodes_buffer.data() + tree_idx,additive_trees[m][k]->nodes.data(),sizeof(Tree::TreeNode) * (2 * config->tree_max_n_leaves - 1));
    }
  }
  cudaMemcpy(d_nodes,nodes_buffer.data(),sizeof(Tree::TreeNode) * M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1),cudaMemcpyHostToDevice);
  
}

void MartGPU::copyBackTrees(int M){
  std::vector<Tree::TreeNode> nodes_buffer(M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
  cudaMemcpy(nodes_buffer.data(),d_nodes,sizeof(Tree::TreeNode) * M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1),cudaMemcpyDeviceToHost);
  
  int K = (data->data_header.n_classes == 2) ? 1 : data->data_header.n_classes;
  for(int m = 0;m < M;++m){
    for(int k = 0;k < K;++k){
      if(additive_trees[m][k] == NULL){
        Tree* tree = new Tree(data, config);
        additive_trees[m][k] = std::unique_ptr<Tree>(tree);
        const int tree_idx = m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1) + k * (2 * config->tree_max_n_leaves - 1);
        memcpy(additive_trees[m][k]->nodes.data(),nodes_buffer.data() + tree_idx,sizeof(Tree::TreeNode) * (2 * config->tree_max_n_leaves - 1));
      }
    }
  }
}

void MartGPU::train() {
  // build one tree if it is binary prediction
  int K = (data->data_header.n_classes == 2) ? 1 : data->data_header.n_classes;

  printf("Using GPU ...\n");

  Utils::Timer t1, t2, t3;
  t1.restart();
  t2.restart();
  t3.restart();

  for (int m = start_epoch; m < config->model_n_iterations; m++) {

    computeHessianResidual();
    buildTrees(m,K);
    if (data->data_header.n_classes == 2) {
      updateF1Kernel<<<NUM_BLOCK,NUM_THREAD>>>(d_F,data->n_data);
    }
    computeLoss();
    double loss = 0;
    cudaMemcpy(&loss,d_res,sizeof(double),cudaMemcpyDeviceToHost);
    int err = computeErr();
    GradientBoostingGPU::print_train_message(m + 1,loss,t1.get_time_restart());
    if (config->save_model && (m + 1) % config->model_save_every == 0){
      copyBackTrees(m + 1);
      saveModel(m + 1);
    }
    if(loss < config->stop_tolerance){
      config->model_n_iterations = m + 1;
      break;
    }
  }
  cudaDeviceSynchronize();
  printf("Training takes %.5f seconds\n", t2.get_time());
  if (config->save_model){
    copyBackTrees(config->model_n_iterations);
    saveModel(config->model_n_iterations);
  }
  if (config->save_importance){
    if (config->save_model == false){
      copyBackTrees(config->model_n_iterations);
    }
    for(int m = 0;m < config->model_n_iterations;++m){
      for(int k = 0;k < K;++k){
        additive_trees[m][k]->feature_importance = &feature_importance;
        additive_trees[m][k]->updateFeatureImportance(m);
      }
    }
    getTopFeatures();
  }
  cudaDeviceSynchronize();
}

//----------Mart end----------//

//----------Regression begin----------//

RegressionGPU::RegressionGPU(Data *data, Config *config) : GradientBoostingGPU(data, config) {
  p_g_allocator = reinterpret_cast<void*>(new cub::CachingDeviceAllocator());
}

/**
 * Method to implement testing process for MART algorithm as described by
 * Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 */
void RegressionGPU::test() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  Utils::Timer t1;
  t1.restart();

  double best_accuracy = 0;
  double low_err = std::numeric_limits<double>::max();
  for (int m = 0; m < config->model_n_iterations; m++) {
    if (additive_trees[m][0] != NULL) {
      additive_trees[m][0]->init(nullptr, &buffer[0], &buffer[1], nullptr,
                                 nullptr, nullptr,ids_tmp.data(),H_tmp.data(),R_tmp.data());
      std::vector<double> updates = additive_trees[m][0]->predictAll(data);
      for (int i = 0; i < data->n_data; i++) {
        F[0][i] += config->model_shrinkage * updates[i];
      }
    }
    if (config->model_mode == "test_rank") {
    } else {
      if ((m + 1) % config->model_eval_every == 0){
        print_test_message(m + 1,t1.get_time_restart(),low_err);
      }
    }
  }

  cudaMemcpy(d_F,F[0].data(),sizeof(double) * data->n_data,cudaMemcpyHostToDevice);
  copyTreesToGPU(config->model_n_iterations);
}

/**
 * Method to implement training process for MART algorithm as described
 * by Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 * @param[in] start: index to start training at.
 */



void RegressionGPU::copyTreesToGPU(int M){
  std::vector<Tree::TreeNode> nodes_buffer(M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
  
  for(int m = 0;m < M;++m){
    const int tree_idx = m * (2 * config->tree_max_n_leaves - 1);
    memcpy(nodes_buffer.data() + tree_idx,additive_trees[m][0]->nodes.data(),sizeof(Tree::TreeNode) * (2 * config->tree_max_n_leaves - 1));
  }
  cudaMemcpy(d_nodes,nodes_buffer.data(),sizeof(Tree::TreeNode) * M * (2 * config->tree_max_n_leaves - 1),cudaMemcpyHostToDevice);
}

void RegressionGPU::copyBackTrees(int M){
  std::vector<Tree::TreeNode> nodes_buffer(M * (2 * config->tree_max_n_leaves - 1));
  cudaMemcpy(nodes_buffer.data(),d_nodes,sizeof(Tree::TreeNode) * M * (2 * config->tree_max_n_leaves - 1),cudaMemcpyDeviceToHost);
  
  for(int m = 0;m < M;++m){
    if(additive_trees[m][0] == NULL){
      Tree* tree = new Tree(data, config);
      additive_trees[m][0] = std::unique_ptr<Tree>(tree);
      const int tree_idx = m * (2 * config->tree_max_n_leaves - 1);
      memcpy(additive_trees[m][0]->nodes.data(),nodes_buffer.data() + tree_idx,sizeof(Tree::TreeNode) * (2 * config->tree_max_n_leaves - 1));
    }
  }
}

void RegressionGPU::computeLoss(){
  cudaMemsetAsync(d_res,0,sizeof(double));
  if(config->regression_huber_loss){
    computeHuberLossKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,data->n_data,d_F,config->regression_lp_loss,config->huber_delta,reinterpret_cast<double*>(d_res));
  }else{
    computeLpLossKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,data->n_data,d_F,config->regression_lp_loss,reinterpret_cast<double*>(d_res));
  }
}

void RegressionGPU::init(){
  GradientBoostingGPU::init();
  if(config->model_mode == "train" && start_epoch == 0){
    double maxn = std::numeric_limits<double>::min();
    double minn = std::numeric_limits<double>::max();
    double sump = 0;
    const double p = config->regression_lp_loss;
    for (double y : data->Y) {
      if(maxn < y)
        maxn = y;
      if(minn > y)
        minn = y;
      sump += pow(fabs(y),p);
    }
    if(data->Y.size() > 0)
      config->regression_stop_factor = pow(config->regression_stop_factor,config->regression_lp_loss / 2.0) * sump / data->n_data;
    if(config->regression_auto_clip_value && maxn - minn > 0){
      config->tree_clip_value *= maxn - minn;
      //printf("[INFO] tree_clip_value is updated to %f\n",config->tree_clip_value);
    }
  }
}

void RegressionGPU::train() {
  // build one tree if it is binary prediction
  int K = 1;
  printf("Using GPU ...\n");

  Utils::Timer t1, t2, t3;
  t1.restart();
  t2.restart();
  t3.restart();

  for (int m = start_epoch; m < config->model_n_iterations; m++) {

    computeHessianResidual();
    buildTree(m,0);
    computeLoss();
    double loss = 0;
    cudaMemcpy(&loss,d_res,sizeof(double),cudaMemcpyDeviceToHost);
    loss = loss / data->n_data;
    print_train_message(m + 1,loss,t1.get_time_restart());
    if (config->save_model && (m + 1) % config->model_save_every == 0){
      copyBackTrees(m + 1);
      saveModel(m + 1);
    }
    if(loss < config->regression_stop_factor){
      config->model_n_iterations = m + 1;
      break;
    }
  }
  cudaDeviceSynchronize();
  printf("Training takes %.5f seconds\n", t2.get_time());
  if (config->save_model){
    copyBackTrees(config->model_n_iterations);
    saveModel(config->model_n_iterations);
  }
  if (config->save_importance){
    if (config->save_model == false){
      copyBackTrees(config->model_n_iterations);
    }
    for(int m = 0;m < config->model_n_iterations;++m){
      for(int k = 0;k < K;++k){
        additive_trees[m][k]->feature_importance = &feature_importance;
        additive_trees[m][k]->updateFeatureImportance(m);
      }
    }
    getTopFeatures();
  }
  cudaDeviceSynchronize();
}


void MartGPU::computeHessianResidual() {
  computeHessianResidualKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,d_F,d_residuals,d_hessians,data->n_data,data->data_header.n_classes);
}
/**
 * Helper method to compute hessian and residual simultaneously.
 */
void RegressionGPU::computeHessianResidual() {
  computeRegressionHessianResidualKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,d_F,d_residuals,d_hessians,data->n_data,config->regression_lp_loss,config->regression_huber_loss,config->huber_delta);
}


//----------Regression end----------//


void GradientBoostingGPU::trySplit(int m,int k,int x){
  trySplitBatch(m,k,x);
}

void GradientBoostingGPU::trySplitSub(int m,int k,int x,int parent,int sibling){
  const int tree_idx = m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1) + k * (2 * config->tree_max_n_leaves - 1);
  const int n_feats = data->data_header.n_feats;
  
  double* d_r_unobserved = reinterpret_cast<double*>(d_res);
  double* d_h_unobserved = reinterpret_cast<double*>(d_res) + 1;

  trySplitBatchSubKernel<<<BATCH_NUM_BLOCK,BATCH_NUM_THREAD>>>(d_nodes + tree_idx,d_ids,d_Xv_offset,x,d_residuals + k * data->n_data,d_hessians + k * data->n_data,d_hist_count_batch,d_hist_sum_batch,d_hist_weight_batch,d_fids,data->n_sample_feats,n_feats,d_n_bins_per_f,config->data_max_n_bins,config->tree_min_node_size,data->n_data,d_block_max_gain,d_block_split_fi,d_block_split_v,d_r_unobserved,d_h_unobserved,d_in_leaf,d_dense_f,d_Xi_offset,d_unobserved_fv,d_cache_hist_count,d_cache_hist_sum,d_cache_hist_weight,parent,sibling,d_accum_bins_start,d_total_bins);
  trySplitBatchFinalizeKernel<<<1,NUM_THREAD>>>(d_nodes + tree_idx + x,d_block_max_gain,d_block_split_fi,d_block_split_v);
}


void GradientBoostingGPU::trySplitBatch(int m,int k,int x){
  const int tree_idx = m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1) + k * (2 * config->tree_max_n_leaves - 1);
  const int n_feats = data->data_header.n_feats;
  
  double* d_r_unobserved = reinterpret_cast<double*>(d_res);
  double* d_h_unobserved = reinterpret_cast<double*>(d_res) + 1;

  cudaMemset(d_r_unobserved,0,sizeof(double));
  cudaMemset(d_h_unobserved,0,sizeof(double));
  trySplitBatchInitSparseKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + tree_idx,d_ids,x,d_residuals + k * data->n_data,d_hessians + k * data->n_data,d_r_unobserved,d_h_unobserved,d_in_leaf);
  if(config->model_use_logit)
    trySplitBatchKernel<true><<<BATCH_NUM_BLOCK,BATCH_NUM_THREAD>>>(d_nodes + tree_idx,d_ids,d_Xv_offset,x,d_residuals + k * data->n_data,d_hessians + k * data->n_data,d_hist_count_batch,d_hist_sum_batch,d_hist_weight_batch,d_fids,data->n_sample_feats,n_feats,d_n_bins_per_f,config->data_max_n_bins,config->tree_min_node_size,data->n_data,d_block_max_gain,d_block_split_fi,d_block_split_v,d_r_unobserved,d_h_unobserved,d_in_leaf,d_dense_f,d_Xi_offset,d_unobserved_fv,d_cache_hist_count,d_cache_hist_sum,d_cache_hist_weight,d_accum_bins_start,d_total_bins);
  else
    trySplitBatchKernel<false><<<BATCH_NUM_BLOCK,BATCH_NUM_THREAD>>>(d_nodes + tree_idx,d_ids,d_Xv_offset,x,d_residuals + k * data->n_data,d_hessians + k * data->n_data,d_hist_count_batch,d_hist_sum_batch,d_hist_weight_batch,d_fids,data->n_sample_feats,n_feats,d_n_bins_per_f,config->data_max_n_bins,config->tree_min_node_size,data->n_data,d_block_max_gain,d_block_split_fi,d_block_split_v,d_r_unobserved,d_h_unobserved,d_in_leaf,d_dense_f,d_Xi_offset,d_unobserved_fv,d_cache_hist_count,d_cache_hist_sum,d_cache_hist_weight,d_accum_bins_start,d_total_bins);
  resetInLeafKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + tree_idx,d_ids,x,d_in_leaf);
  trySplitBatchFinalizeKernel<<<1,NUM_THREAD>>>(d_nodes + tree_idx + x,d_block_max_gain,d_block_split_fi,d_block_split_v);
}

template<class T>
void generate_sequence(T* d_data,int n,int sample_len,int seed,cub::CachingDeviceAllocator& g_allocator){
  sequenceKernel<<<nB,nT>>>(d_data,n);
  if(n != sample_len){
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,seed);
    void* d_rand_void = NULL;
    g_allocator.DeviceAllocate(&d_rand_void,sizeof(float) * n);
    float* d_rand = reinterpret_cast<float*>(d_rand_void);

    curandGenerateUniform(gen,d_rand,n);
    void     *d_tmp = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_tmp, temp_storage_bytes,
        d_rand, d_rand, d_data, d_data, n);
        // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
        // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_tmp, temp_storage_bytes,
        d_rand, d_rand, d_data, d_data, n);
    g_allocator.DeviceFree(d_tmp);
    g_allocator.DeviceFree(d_rand_void);
    
    d_tmp = NULL;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_tmp, temp_storage_bytes, d_data, d_data, sample_len);
        // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(d_tmp, temp_storage_bytes, d_data, d_data, sample_len);
    g_allocator.DeviceFree(d_tmp);
  }
}

void GradientBoostingGPU::buildTree(int m,int k){
  cub::CachingDeviceAllocator&  g_allocator = *reinterpret_cast<cub::CachingDeviceAllocator*>(p_g_allocator); 
  int n_sample_data = (config->model_data_sample_rate < 1 - 1e-9) ? config->model_data_sample_rate * data->n_data : data->n_data;
  generate_sequence(d_ids,data->n_data,n_sample_data,m + 1,g_allocator);
  data->n_sample_feats = (config->model_feature_sample_rate < 1 - 1e-9) ? config->model_feature_sample_rate * data->data_header.n_feats : data->data_header.n_feats;
  generate_sequence(d_fids,data->data_header.n_feats,data->n_sample_feats,m + 1 + 1000000,g_allocator);
  
  int tree_idx = m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1) + k * (2 * config->tree_max_n_leaves - 1);
  buildTreeStartKernel<<<1,1>>>(d_nodes,tree_idx,n_sample_data);
  trySplit(m, k, 0);
  for (int i = 0; i < config->tree_max_n_leaves - 1; ++i) {
    // find the node with max gain to split (calculated in trySplit)
    findSplitIdxKernel<<<1,1>>>(d_nodes + tree_idx,i,reinterpret_cast<int*>(d_res));
    // Split Begin
    int l = 2 * i + 1;
    int r = l + 1;

    setFeatureIdxPxKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + tree_idx,d_ids,reinterpret_cast<int*>(d_res),d_in_leaf,d_Xi_offset,d_feature_id2idx);
    splitSetFlagKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + tree_idx,reinterpret_cast<int*>(d_res),l,d_Xv_offset,data->n_data,d_ids,d_temp_storage,d_flags,d_dense_f,d_unobserved_fv,d_feature_id2idx);
    resetFeatureIdxPxKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + tree_idx,d_ids,reinterpret_cast<int*>(d_res),d_in_leaf,d_Xi_offset,d_feature_id2idx);

    int node_info[3];
    cudaMemcpy(node_info,d_res,sizeof(int) * 3,cudaMemcpyDeviceToHost);
    int split_x = node_info[0];
    int start = node_info[1];
    int num_items = node_info[2];

    // Determine temporary device storage requirements for inclusive prefix sum
    void     *d_tmp = NULL;
    size_t   temp_storage_bytes = 0;
    int* d_num_selected_out = reinterpret_cast<int*>(d_res);
    cub::DevicePartition::Flagged(d_tmp, temp_storage_bytes, d_ids + start, d_flags + start, d_prefix_sum, d_num_selected_out, num_items);
    // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
    // Run selection
    cub::DevicePartition::Flagged(d_tmp, temp_storage_bytes, d_ids + start, d_flags + start, d_prefix_sum + start, d_num_selected_out, num_items);
    g_allocator.DeviceFree(d_tmp);
    cudaMemcpy(node_info,d_res,sizeof(int) * 3,cudaMemcpyDeviceToHost);
    int num_selected = node_info[0];
    d_tmp = NULL;
    cub::DeviceRadixSort::SortKeys(d_tmp, temp_storage_bytes, d_prefix_sum + start + num_selected, d_ids + start + num_selected, num_items - num_selected);
    // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(d_tmp, temp_storage_bytes, d_prefix_sum + start + num_selected, d_ids + start + num_selected, num_items - num_selected);
    g_allocator.DeviceFree(d_tmp);
    d_tmp = NULL;
    cub::DeviceRadixSort::SortKeys(d_tmp, temp_storage_bytes, d_prefix_sum + start , d_ids + start, num_selected);
    // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(d_tmp, temp_storage_bytes, d_prefix_sum + start , d_ids + start, num_selected);
    g_allocator.DeviceFree(d_tmp);
    d_tmp = NULL;
    splitFinalizeKernel<<<1,1>>>(d_nodes + tree_idx,l,num_selected);
    //Split End

    int lsz = num_selected, rsz = num_items - lsz;
    if(i + 1 < config->tree_max_n_leaves - 1){
      if(lsz <= rsz){
        trySplit(m,k,l);
        //trySplit(m,k,r); is replaced with the following quick substraction
        trySplitSub(m,k,r,split_x,l);
      }else{
        trySplit(m,k,r);
        //trySplit(m,k,l);
        trySplitSub(m,k,l,split_x,r);
      }
    }
  }
  double correction = 1.0;
  if (data->data_header.n_classes != 1 && config->model_name.size() >= 3 &&
      config->model_name.substr(0, 3) != "abc")
    correction -= 1.0 / data->data_header.n_classes;
  for(int i = 0;i < (2 * config->tree_max_n_leaves - 1);++i){
    regressKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + tree_idx,i, (2 * config->tree_max_n_leaves - 1),d_ids,d_residuals + k * data->n_data,d_hessians + k * data->n_data,correction,config->tree_clip_value,config->tree_damping_factor,d_sum_numerator,d_sum_denominator);


    void     *d_tmp = NULL;
    size_t   temp_storage_bytes = 0;
    double* d_numerator = reinterpret_cast<double*>(d_res);
    double* d_denominator = reinterpret_cast<double*>(d_res) + 1;
    cub::DeviceReduce::Sum(d_tmp, temp_storage_bytes, d_sum_numerator, d_numerator, NUM_BLOCK * NUM_THREAD);
    // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Sum(d_tmp, temp_storage_bytes, d_sum_numerator, d_numerator, NUM_BLOCK * NUM_THREAD);
    g_allocator.DeviceFree(d_tmp);
    d_tmp = NULL;
    cub::DeviceReduce::Sum(d_tmp, temp_storage_bytes, d_sum_denominator, d_denominator, NUM_BLOCK * NUM_THREAD);
    // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Sum(d_tmp, temp_storage_bytes, d_sum_denominator, d_denominator, NUM_BLOCK * NUM_THREAD);
    g_allocator.DeviceFree(d_tmp);
    d_tmp = NULL;

    regressFinalizeKernel<<<1,1>>>(d_nodes + tree_idx,i, correction,config->tree_clip_value,config->tree_damping_factor,d_numerator,d_denominator);
  }

  updateFKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + tree_idx, (2 * config->tree_max_n_leaves - 1),d_F + k * data->n_data, d_ids,config->model_shrinkage);
}

void MartGPU::buildTrees(int m,int K){
  for(int k = 0;k < K;++k){
    buildTree(m,k);
  }
}


void MartGPU::computeLoss(){
  cudaMemsetAsync(d_res,0,sizeof(double));
  computeLossKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,data->n_data,d_F,data->data_header.n_classes,reinterpret_cast<double*>(d_res));
}

//----------Mart end----------//

//----------ABCMart begin----------//

ABCMartGPU::ABCMartGPU(Data *data, Config *config) : GradientBoostingGPU(data, config) {
  p_g_allocator = reinterpret_cast<void*>(new cub::CachingDeviceAllocator());
}


void ABCMartGPU::init(){
  GradientBoostingGPU::init();

  base_classes.resize(config->model_n_iterations, 0);
  class_losses.resize(data->data_header.n_classes, 0.0);
  cudaMalloc(&d_class_loss,sizeof(double) * data->data_header.n_classes);
  initBaseClass();

  cudaDeviceSynchronize();
}

__global__
void baseClassInitKernel(int* d_res,int* d_histogram,int size,double* d_class_loss){
  int maxCount = 1;
  int res = -1;
  for (int i = 0; i < size; ++i) {
    if (d_histogram[i] > maxCount) {
      maxCount = d_histogram[i];
      res = i;
    }
    d_class_loss[i] = d_histogram[i];
  }
  *d_res = res;
}

void ABCMartGPU::initBaseClass(){
  cub::CachingDeviceAllocator&  g_allocator = *reinterpret_cast<cub::CachingDeviceAllocator*>(p_g_allocator); 
  int      num_samples = data->Y.size();
  void* d_histogram_void;
  g_allocator.DeviceAllocate(&d_histogram_void, data->data_header.n_classes * sizeof(int));
  int*     d_histogram = (int*)d_histogram_void;
  int      num_levels = data->data_header.n_classes;
  float    lower_level = 0;    // e.g., 0.0     (lower sample value boundary of lowest bin)
  float    upper_level = data->data_header.n_classes - 1;
  // Determine temporary device storage requirements
  void*    d_tmp = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceHistogram::HistogramEven(d_tmp, temp_storage_bytes, d_Y, d_histogram, num_levels, lower_level, upper_level, num_samples);
  // Allocate temporary storage
  g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramEven(d_tmp, temp_storage_bytes, d_Y, d_histogram, num_levels, lower_level, upper_level, num_samples);
  g_allocator.DeviceFree(d_tmp);

  baseClassInitKernel<<<1,1>>>(reinterpret_cast<int*>(d_res),d_histogram,num_levels,d_class_loss);
  cudaMemcpy(&base_classes[0],d_res,sizeof(int),cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  g_allocator.DeviceFree(d_histogram);
}

/**
 * Method to implement testing process for MART algorithm as described by
 * Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 */
void ABCMartGPU::test() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  Utils::Timer t1;
  t1.restart();

  double best_accuracy = 0;
  int K = data->data_header.n_classes;
  for (int m = 0; m < config->model_n_iterations; ++m) {
    printf("%d: %d\n",m,base_classes[m]);
    for (int k = 0; k < K; ++k) {
      int b = base_classes[m];
      if (additive_trees[m][k] != NULL) {
        additive_trees[m][k]->init(nullptr, &buffer[0], &buffer[1], nullptr,
                                   nullptr, nullptr,
                                   nullptr,nullptr,nullptr);
        std::vector<double> updates = additive_trees[m][k]->predictAll(data);
        for (int i = 0; i < data->n_data; ++i) {
          F[k][i] += config->model_shrinkage * updates[i];
          F[b][i] -= config->model_shrinkage * updates[i];
        }
      }
    }

    double acc = getAccuracy();
    if (acc > best_accuracy) best_accuracy = acc;
    if ((m + 1) % config->model_eval_every == 0)
      printf("%d\tTime: %f | acc/best_acc: %f/%f | loss: %f \n", m + 1,
             t1.get_time_restart(), acc, best_accuracy, getLoss());
  }
  for(int k = 0;k < data->data_header.n_classes;++k)
    cudaMemcpy(d_F + k * data->n_data,F[k].data(),sizeof(double) * data->n_data,cudaMemcpyHostToDevice);
  copyTreesToGPU(config->model_n_iterations);
}

/**
 * Method to implement training process for MART algorithm as described
 * by Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 * @param[in] start: index to start training at.
 */



void ABCMartGPU::copyTreesToGPU(int M){
  std::vector<Tree::TreeNode> nodes_buffer(M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
  
  int K = data->data_header.n_classes;
  for(int m = 0;m < M;++m){
    for(int k = 0;k < K;++k){
      if(base_classes[m] != k){
        const int tree_idx = m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1) + k * (2 * config->tree_max_n_leaves - 1);
        memcpy(nodes_buffer.data() + tree_idx,additive_trees[m][k]->nodes.data(),sizeof(Tree::TreeNode) * (2 * config->tree_max_n_leaves - 1));
      }
    }
  }
  cudaMemcpy(d_nodes,nodes_buffer.data(),sizeof(Tree::TreeNode) * M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1),cudaMemcpyHostToDevice);
}


void ABCMartGPU::copyBackTrees(int M){
  std::vector<Tree::TreeNode> nodes_buffer(M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
  cudaMemcpy(nodes_buffer.data(),d_nodes,sizeof(Tree::TreeNode) * M * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1),cudaMemcpyDeviceToHost);
  
  int K = data->data_header.n_classes;
  for(int m = 0;m < M;++m){
    for(int k = 0;k < K;++k){
      if(additive_trees[m][k] == NULL && base_classes[m] != k){
        Tree* tree = new Tree(data, config);
        additive_trees[m][k] = std::unique_ptr<Tree>(tree);
        const int tree_idx = m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1) + k * (2 * config->tree_max_n_leaves - 1);
        memcpy(additive_trees[m][k]->nodes.data(),nodes_buffer.data() + tree_idx,sizeof(Tree::TreeNode) * (2 * config->tree_max_n_leaves - 1));
      }
    }
  }

  cudaMemcpy(class_losses.data(),d_class_loss,sizeof(double) * data->data_header.n_classes,cudaMemcpyDeviceToHost);
}

void ABCMartGPU::train() {
  printf("Using GPU ...\n");
  
  int n_skip = 1;
  if(config->model_gap != ""){
    n_skip = atoi(config->model_gap.c_str()) + 1;
    return exhaustiveTrain(n_skip);
  }

  int K = data->data_header.n_classes;


  Utils::Timer t1, t2, t3;
  t1.restart();
  t2.restart();
  t3.restart();

  if(start_epoch != 0){
    base_classes[start_epoch] = argmax(class_losses);
  }
  for (int m = start_epoch; m < config->model_n_iterations; m++) {
    computeHessianResidual(m);

    buildTrees(m,K);
    computeLoss(m);
    printKernel<<<1,1>>>(reinterpret_cast<double*>(d_res),1);
    double loss = 0;
    cudaMemcpy(&loss,d_res,sizeof(double),cudaMemcpyDeviceToHost);
  

    if (config->save_model && (m + 1) % config->model_save_every == 0){
      copyBackTrees(m + 1);
      saveModel(m + 1);
    }
    if(loss < config->stop_tolerance){
      config->model_n_iterations = m + 1;
      break;
    }
  }
  cudaDeviceSynchronize();
  printf("Training takes %.5f seconds\n", t2.get_time());
  if (config->save_model){
    copyBackTrees(config->model_n_iterations);
    saveModel(config->model_n_iterations);
  }
  if (config->save_importance){
    if (config->save_model == false){
      copyBackTrees(config->model_n_iterations);
    }
    for(int m = 0;m < config->model_n_iterations;++m){
      for(int k = 0;k < K;++k){
        if(k == base_classes[m])
          continue;
        additive_trees[m][k]->feature_importance = &feature_importance;
        additive_trees[m][k]->updateFeatureImportance(m);
      }
    }
    getTopFeatures();
  }
  cudaDeviceSynchronize();
}



/**
 * Helper method to compute hessian and residual simultaneously.
 */
void ABCMartGPU::computeHessianResidual() {
  computeHessianResidualKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,d_F,d_residuals,d_hessians,data->n_data,data->data_header.n_classes);
}

void ABCMartGPU::computeHessianResidual(int m) {
  computeHessianResidualABCKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,d_F,d_residuals,d_hessians,data->n_data,data->data_header.n_classes,base_classes[m]);
}

void ABCMartGPU::computeHessianResidual(int m,double* d_prob) {
  computeHessianResidualABCStaticMallocKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,d_F,d_residuals,d_hessians,data->n_data,data->data_header.n_classes,base_classes[m],d_prob);
}


__global__
void updateBaseFValuesKernel(double* d_F, int n_data, int n_classes,int b){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for (int i = gid; i < n_data; i += step) {
    double base_f = 0.0;
    for (int k = 0; k < n_classes; ++k) {
      if (k == b) continue;  // skip base class, as value is being calculated
      base_f -= d_F[k * n_data + i];
    }
    d_F[b * n_data + i] = base_f;
  }
}

void ABCMartGPU::updateBaseFValues(int b) {
  updateBaseFValuesKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_F,data->n_data,data->data_header.n_classes,b);
}

void ABCMartGPU::buildTrees(int m,int K){
  int b = base_classes[m];
  for(int k = 0;k < K;++k){
    if (k == b) {
      additive_trees[m][k] = std::unique_ptr<Tree>(nullptr);
      continue;
    }
    buildTree(m,k);
  }
  updateBaseFValues(b);
}


void ABCMartGPU::computeLoss(int m){
  cudaMemsetAsync(d_res,0,sizeof(double));
  cudaMemsetAsync(d_class_loss,0,sizeof(double) * data->data_header.n_classes);
  computeLossAndClassLossKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,data->n_data,d_F,data->data_header.n_classes,d_class_loss,reinterpret_cast<double*>(d_res));

  cub::CachingDeviceAllocator&  g_allocator = *reinterpret_cast<cub::CachingDeviceAllocator*>(p_g_allocator); 
  int num_items = data->data_header.n_classes;
  void *d_out_void;
  g_allocator.DeviceAllocate(&d_out_void, sizeof(cub::KeyValuePair<int, double>));
  cub::KeyValuePair<int, double>   *d_out;
  d_out = (cub::KeyValuePair<int, double>*)d_out_void;

  // Determine temporary device storage requirements
  void     *d_tmp = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceReduce::ArgMax(d_tmp, temp_storage_bytes, d_class_loss, d_out, num_items);
  // Allocate temporary storage
  g_allocator.DeviceAllocate(&d_tmp, temp_storage_bytes);
  // Run argmax-reduction
  cub::DeviceReduce::ArgMax(d_tmp, temp_storage_bytes, d_class_loss, d_out, num_items);

  cub::KeyValuePair<int, double> h_out;
  cudaMemcpy(&h_out,d_out,sizeof(cub::KeyValuePair<int, double>),cudaMemcpyDeviceToHost);
  if(m + 1 < config->model_n_iterations){
    base_classes[m + 1] = h_out.key;
  }

  g_allocator.DeviceFree(d_tmp);
  g_allocator.DeviceFree(d_out);

}

void ABCMartGPU::saveModel(int iter) {
  FILE *model_out =
      fopen((experiment_path + config->model_suffix).c_str(), "w");
  if (model_out == NULL) {
    fprintf(stderr, "[ERROR] Cannot create file: (%s)\n",
            (experiment_path + config->model_suffix).c_str());
    exit(1);
  }
  ModelHeader model_header;
  model_header.config = *config;
  model_header.config.model_n_iterations = iter;

  model_header.serialize(model_out);
  serializeTrees(model_out, iter);
  // save trees
  for (int i = 0; i < iter; ++i) {
    fwrite(&(base_classes[i]), sizeof(int), 1, model_out);
  }
  Utils::serialize_vector(model_out, class_losses);
  fclose(model_out);
}

int ABCMartGPU::loadModel() {
  FILE *fp = fopen(config->model_pretrained_path.c_str(), "rb");
  if (fp == NULL) {
    return -1;
  }
  // retrieve trees
  ModelHeader model_header = ModelHeader::deserialize(fp);
  GradientBoosting::deserializeTrees(fp);
  base_classes.resize(config->model_n_iterations);

  int size = std::max(config->model_n_iterations,
                      model_header.config.model_n_iterations);
  base_classes.resize(size, 0);

  for (int b = 0; b < model_header.config.model_n_iterations; ++b) {
    auto ret = fread(&base_classes[b], sizeof(int), 1, fp);
  }
  class_losses = Utils::deserialize_vector<double>(fp);

  fclose(fp);
  return 0;
}

__global__
void memcpyKernel(void* dst,void* src,long long size){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  char* cdst = (char*)dst;
  char* csrc = (char*)src;
  const int step = blockDim.x * gridDim.x;
  for (int i = gid; i < size; i += step) {
    cdst[i] = csrc[i];
  }
}

int GradientBoostingGPU::computeErr(){
  cudaMemsetAsync(d_res,0,sizeof(int));
  computeErrorKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_Y,data->n_data,d_F,data->data_header.n_classes,reinterpret_cast<int*>(d_res));
  int err = 0;
  cudaMemcpy(&err,d_res,sizeof(int),cudaMemcpyDeviceToHost);
  return err;
}

void GradientBoostingGPU::print_train_message(int iter,double loss,double iter_time){
  int err = computeErr();
  printf("%4d | loss: %20.14e | errors: %7d | time: %.5f\n", iter,
       loss, err, iter_time);
  fprintf(log_out,"%4d %20.14e %7d %.5f\n", iter,
       loss, err, iter_time);
}

void RegressionGPU::print_train_message(int iter,double loss,double iter_time){
  printf("%4d | loss: %20.14e | time: %.5f\n", iter,
       loss, iter_time);
  fprintf(log_out,"%4d %20.14e %.5f\n", iter,
       loss, iter_time);
}

__global__
void normalizeFKernel(double* d_F,int m,int K,int n){
  int tid = threadIdx.x, bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  for(long long i = gid;i < n;i += step){
    double sum = 0;
    for (int k = 0;k < K;++k){
      sum += d_F[k * n + i];
    }
    sum /= K;
    for (int k = 0;k < K;++k){
      d_F[k * n + i] -= sum;
    }
  }
}

void ABCMartGPU::normalizeF(int m,int K,int n){
  normalizeFKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_F,m,K,n);
}

void ABCMartGPU::exhaustiveTrain(int n_skip) {
  int K = data->data_header.n_classes;

  double* d_backup_F = NULL;
  double* d_best_F = NULL;
  cudaMalloc(&d_backup_F,sizeof(double) * data->data_header.n_classes * data->n_data);
  cudaMalloc(&d_best_F,sizeof(double) * data->data_header.n_classes * data->n_data);
  Tree::TreeNode* d_best_nodes = NULL;
  cudaMalloc(&d_best_nodes,sizeof(Tree::TreeNode) * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1)); 
  double* d_prob = NULL;
  cudaMalloc(&d_prob,sizeof(double) * data->data_header.n_classes * NUM_BLOCK * NUM_THREAD);

  Utils::Timer t1, t2;
  t1.restart();
  t2.restart();
  std::vector<int> base_candidates;
  if(config->base_candidates_size == 0 || config->base_candidates_size == data->data_header.n_classes){
    for(int i = 0;i < data->data_header.n_classes;++i)
      base_candidates.push_back(i);
  }
  for (int m = start_epoch; m < config->model_n_iterations; ++m) {
    double current_loss = 0;
    if(m < config->warmup_iter){
      computeHessianResidual();
      for(int k = 0;k < K;++k){
        buildTree(m,k);
      }
      computeLoss(m);
      double loss = 0;
      cudaMemcpy(&loss,d_res,sizeof(double),cudaMemcpyDeviceToHost);
      current_loss = loss;
      int err = computeErr();
      GradientBoostingGPU::print_train_message(m + 1,loss,t1.get_time_restart());
    }else if(config->base_candidates_size <= 1 && (m - config->warmup_iter) % n_skip == 0){
      if(config->warmup_iter > 0 && m == config->warmup_iter){
        normalizeF(m,K,data->n_data);
      }
      computeHessianResidual(m,d_prob);
      buildTrees(m,K);
      computeLoss(m);
      double loss = 0;
      cudaMemcpy(&loss,d_res,sizeof(double),cudaMemcpyDeviceToHost);
      current_loss = loss;
      int err = computeErr();
      GradientBoostingGPU::print_train_message(m + 1,loss,t1.get_time_restart());
    
    }else if(config->base_candidates_size > 1 && (m - config->warmup_iter) % n_skip == 0){  
      if(config->warmup_iter > 0 && m == config->warmup_iter){
        normalizeF(m,K,data->n_data);
      }
      memcpyKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_backup_F,d_F,sizeof(double) * data->data_header.n_classes * data->n_data);
      double best_base_loss = std::numeric_limits<double>::max();
      int best_base = 0;
      if(config->base_candidates_size != 0 && config->base_candidates_size != data->data_header.n_classes){
        cudaMemcpy(class_losses.data(),d_class_loss,sizeof(double) * data->data_header.n_classes,cudaMemcpyDeviceToHost);
        std::vector<int> loss_idx(class_losses.size());
        for(int i = 0;i < loss_idx.size();++i)
          loss_idx[i] = i;
        std::sort(loss_idx.begin(),loss_idx.end(),[&](int a,int b){
              if(class_losses[a] != class_losses[b])
                return class_losses[a] > class_losses[b];
              return data->data_header.idx2label[a] < data->data_header.idx2label[b];
            });
        base_candidates.clear();
        for(int i = 0;i < config->base_candidates_size && i < data->data_header.n_classes;++i)
          base_candidates.push_back(loss_idx[i]);
      }
      for (int b : base_candidates) {
        base_classes[m] = b;
        computeHessianResidual(m,d_prob);
        buildTrees(m,K);
        computeLoss(m);
        double loss = 0;
        cudaMemcpy(&loss,d_res,sizeof(double),cudaMemcpyDeviceToHost);
        if(loss < best_base_loss){
          best_base_loss = loss;
          best_base = b;
          memcpyKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_best_F,d_F,sizeof(double) * data->data_header.n_classes * data->n_data);
          memcpyKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_best_nodes,d_nodes + m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1),sizeof(Tree::TreeNode) * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
        }
        memcpyKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_F,d_backup_F,sizeof(double) * data->data_header.n_classes * data->n_data);
        initNodeKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1),data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
      }
            current_loss = best_base_loss;
      base_classes[m] = best_base;
      memcpyKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_F,d_best_F,sizeof(double) * data->data_header.n_classes * data->n_data);
      memcpyKernel<<<NUM_BLOCK,NUM_THREAD>>>(d_nodes + m * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1),d_best_nodes,sizeof(Tree::TreeNode) * data->data_header.n_classes * (2 * config->tree_max_n_leaves - 1));
      cudaDeviceSynchronize();
      int err = computeErr();
      GradientBoostingGPU::print_train_message(m + 1,best_base_loss,t1.get_time_restart());
    }else{
      base_classes[m] = base_classes[m - 1];
      computeHessianResidual(m,d_prob);
      buildTrees(m,K);
      computeLoss(m);
      double loss = 0;
      cudaMemcpy(&loss,d_res,sizeof(double),cudaMemcpyDeviceToHost);
      current_loss = loss;
      int err = computeErr();
      GradientBoostingGPU::print_train_message(m + 1,loss,t1.get_time_restart());
    }

    if (config->save_model && (m + 1) % config->model_save_every == 0){
      copyBackTrees(m + 1);
      saveModel(m + 1);
    }
    if(current_loss < config->stop_tolerance){
      config->model_n_iterations = m + 1;
      break;
    }
  }
  printf("Training has taken %.5f seconds\n", t2.get_time());
  if (config->save_model){
    copyBackTrees(config->model_n_iterations);
    saveModel(config->model_n_iterations);
  }
  if (config->save_importance){
    if (config->save_model == false){
      copyBackTrees(config->model_n_iterations);
    }
    for(int m = 0;m < config->model_n_iterations;++m){
      for(int k = 0;k < K;++k){
        if(k == base_classes[m])
          continue;
        additive_trees[m][k]->feature_importance = &feature_importance;
        additive_trees[m][k]->updateFeatureImportance(m);
      }
    }
    getTopFeatures();
  }
  cudaDeviceSynchronize();
}

//----------ABCMart end----------//

}  // namespace ABCBoost
