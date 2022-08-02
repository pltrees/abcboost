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

#ifndef ABCBOOST_UTILS_H
#define ABCBOOST_UTILS_H

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
#define OS_WIN
#endif

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef OS_WIN
#define strtok_r strtok_s
#include <direct.h>
#else
#include <unistd.h>
#endif
#include <array>
#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <stdarg.h>
#include<cmath>
#include "config.h"


#ifdef CUDA
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <iostream>
#endif

#define CONDITION_OMP_PARALLEL_FOR(prag, condition, ...) if(condition) {_Pragma(#prag) __VA_ARGS__} else {__VA_ARGS__}

#define DEBUG_PRINT
inline int debug_printf ( const char * format, ... ){
	#ifdef DEBUG_PRINT
		va_list args;
		va_start(args, format);
		auto ret = vprintf(format, args);
		va_end(args);
		return ret;
	#else
		return 0;
	#endif
}

namespace ABCBoost {

//=============================================================================
//
// Meta data types
//
//=============================================================================
typedef unsigned int uint;
typedef unsigned short ushort;
typedef double hist_t;

typedef double gpu_numeric_t;

struct GPUArgs {
  int *d_bc;
  float *d_bs, *d_bw, *d_H, *d_R;
  uint *d_ids;
//  ushort *d_Xv;
  // below is for multi-GPU computation
#ifdef CUDA
  cudaStream_t stream;
  double* d_Y; 
  unsigned int* d_Xi;    // store the instance ids
  unsigned short* d_Xv;  // store the corresponding value
  double* d_F;
  double* d_hessians;
  double* d_residuals;

#endif
  uint f_start, f_end, device_id;
};

namespace Utils {

//=============================================================================
//
// CPU utility functions
//
//=============================================================================

template <class T>
inline void serialize(FILE *fp, T &x) {
  fwrite(&x, sizeof(T), 1, fp);
}

template <>
inline void serialize(FILE *fp, std::string &x) {
  size_t size = x.length();
  fwrite(&size, sizeof(size), 1, fp);
  fwrite(x.c_str(), size, 1, fp);
}

template <>
inline void serialize(FILE *fp, const std::string &x) {
  size_t size = x.length();
  fwrite(&size, sizeof(size), 1, fp);
  fwrite(x.c_str(), size, 1, fp);
}

inline std::string deserialize_str(FILE *fp) {
  std::string str;
  size_t size;
  int dummy_return = 0;
  dummy_return = fread(&size, sizeof(size), 1, fp);
  str.resize(size);
  dummy_return = fread(&str[0], size, 1, fp);
  return str;
}

template <class T>
inline void serialize_vector(FILE *fp, std::vector<T> &x) {
  size_t size = x.size();
  fwrite(&size, sizeof(size), 1, fp);
  fwrite(x.data(), size, sizeof(T), fp);
}

template <class T>
inline std::vector<T> deserialize_vector(FILE *fp) {
  std::vector<T> vec;
  size_t size;
  int dummy_return = 0;
  dummy_return = fread(&size, sizeof(size), 1, fp);
  vec.resize(size);
  dummy_return = fread(vec.data(), size, sizeof(T), fp);
  return vec;
}

template <class T>
inline void serialize_vector2d(FILE *fp, std::vector<std::vector<T>> &x) {
  size_t size = x.size();
  fwrite(&size, sizeof(size), 1, fp);
  for (int i = 0; i < size; ++i) serialize_vector(fp, x[i]);
}

template <class T>
inline std::vector<std::vector<T>> deserialize_vector2d(FILE *fp) {
  std::vector<std::vector<T>> vecs;
  size_t size;
  int dummy_return = 0;
  dummy_return = fread(&size, sizeof(size), 1, fp);
  vecs.resize(size);
  for (int i = 0; i < size; ++i) vecs[i] = deserialize_vector<T>(fp);
  return vecs;
}

template <class T>
inline T deserialize(FILE *fp) {
  int ret = 0;
  auto dummy_return = fread(&ret, sizeof(ret), 1, fp);
  return ret;
}

class Timer {
 private:
  std::chrono::high_resolution_clock::time_point st;

 public:
  void restart() { st = std::chrono::high_resolution_clock::now(); }

  double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = now - st;
    return fp_ms.count() / 1000.0;
  }

  double get_time_restart() {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = now - st;
    restart();
    return fp_ms.count() / 1000.0;
  }
};

template <typename Out>
void getTokens(const std::string &s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

inline std::vector<std::string> getTokens(const std::string &s, char delim) {
  std::vector<std::string> elems;
  getTokens(s, delim, std::back_inserter(elems));
  return elems;
}

//=============================================================================
//
// GPU utility functions
//
//=============================================================================

void initGPUArgs(Config *config, std::vector<GPUArgs> &args,
                 std::vector<uint> &fids, std::vector<uint> &valid_fi,
                 std::vector<short> &dense_f,
                 std::vector<std::vector<ushort>> &Xv, int nB, int n_data);

void updateGPUArgs(std::vector<GPUArgs> &args, std::vector<double> &H,
                   std::vector<double> &R);

void freeDeviceMem(std::vector<GPUArgs> &args);

class RankUtils{
public:
  const int NDCG_COUNT_ZERO_ONE = 1;
  
  // return -1 when maxDCG = 0
  // in that case, we eliminate the query groups whose maxDCG equals 0
  //template<class T,class U>
  static double computeNDCG(const std::vector<double>& pred_score,const std::vector<double>& label_score,int label_start){
    std::vector<int> idx(pred_score.size());
    for(int i = 0;i < idx.size();++i)
      idx[i] = i;
    auto idx_max = idx;
    std::sort(idx.begin(),idx.end(),[&pred_score](int a,int b){
        return pred_score[a] > pred_score[b];});
    std::sort(idx_max.begin(),idx_max.end(),[&label_score,label_start](int a,int b){
        return label_score[a + label_start] > label_score[b + label_start];});
    double DCG = 0;
    for(int i = 0;i < idx.size();++i)
      DCG += ((1 << (int)label_score[label_start + idx[i]]) - 1) / log2(2.0 + i);
    double maxDCG = 0;
    for(int i = 0;i < idx_max.size();++i)
      maxDCG += ((1 << (int)label_score[label_start + idx_max[i]]) - 1) / log2(2.0 + i);
    
    if(maxDCG < 1e-10)
      return -1;
    return DCG / maxDCG;
  }
};


}  // namespace Utils

}  // namespace ABCBoost

#endif  // ABCBOOST_UTILS_H
