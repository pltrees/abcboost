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

#include "utils.h"
#ifdef OMP
#include "omp.h"
#else
#include "dummy_omp.h"
#endif
namespace ABCBoost {

namespace Utils {

/**
 * Allocate device memory and copy Xv only once. 
 * @param[in] config: Pointer to configuration
 *            args: Vector of GPU arguments
 *            fids: Sampled feature ids
 *            valid_fi: Non-empty feature ids
 *            dense_f: Vector indicating whether features are dense or sparse
 *            Xv: Feature values
 *            nB: Number of blocks
 *            n_data: Number of instances
*/
void initGPUArgs(
  Config *config, 
  std::vector<GPUArgs> &args,
  std::vector<uint> &fids, 
  std::vector<uint> &valid_fi,
  std::vector<short> &dense_f, 
  std::vector<std::vector<ushort>> &Xv, 
  int nB, int n_data) {

  uint n_samples = n_data * config->model_data_sample_rate;
  
  int n_dense = 0;
  uint fsz = fids.size();
  for (int j = 0; j < fsz; ++j) { 
    int fid = valid_fi[fids[j]];
    if (dense_f[fid])  n_dense++;
  }

  int n_devices;
  cudaGetDeviceCount(&n_devices);
  std::vector<int> device_ids;
  for (int i = 0; i < n_devices; ++i) {
    cudaSetDevice(i);
    size_t free_t, total_t;
    cudaMemGetInfo(&free_t, &total_t);
    if (free_t / (double) total_t > 0.5) {
      device_ids.push_back(i);
    }
  }

  int n_available_devices = (n_dense > device_ids.size()) ?
                            device_ids.size() : n_dense;

  args = std::vector<GPUArgs>(n_available_devices);
  for (int i = 0; i < n_available_devices; ++i) {
    cudaSetDevice(device_ids[i]);
    cudaStreamCreate(&(args[i].stream));
  }
  
  printf("Using %d (out of %d) GPUs\n", n_available_devices, n_devices);
  
  uint step;
  if (n_dense == 0) {
    step = 0;
    printf("No dense features for GPU computing.\n");
  } else if (n_available_devices == 0) {
    printf("No available GPUs.\n");
    exit(1);
  } else {
    step = (n_dense + n_available_devices - 1) / n_available_devices;
  }

  uint j = 0; 
  uint last_f_end = 0;
  for (int i = 0; i < n_available_devices; ++i) {
		int *d_bc;
    float *d_bw, *d_bs, *d_H, *d_R;
    uint *d_ids;
    ushort *d_Xv;

    cudaSetDevice(device_ids[i]);
    
    uint f_start = last_f_end;
    // make sure there is at least one feature for following uninitialized GPUs
    uint f_end = (f_start + step + (n_available_devices - i - 1) < n_dense) ? f_start + step + (n_available_devices - i - 1) : n_dense;
    last_f_end = f_end;
 
    uint64_t sz = sizeof(float) * (f_end - f_start) * config->data_max_n_bins * nB;
    uint64_t sz_int = sizeof(int) * (f_end - f_start) * config->data_max_n_bins * nB;
    // malloc GPU memory
    cudaMalloc((void**) &d_bc, sz_int);
    cudaMalloc((void**) &d_bs, sz);
    cudaMalloc((void**) &d_bw, sz);

    sz = sizeof(float) * n_data;
    cudaMalloc((void**) &d_H, sz); 
    cudaMalloc((void**) &d_R, sz); 
    cudaMalloc((void**) &d_ids, sizeof(uint) * n_samples);
    cudaMalloc((void**) &d_Xv, sizeof(ushort) * (f_end - f_start) * n_data);

    // copy dense features to device memory
    ushort *device = d_Xv;
    int count = 0;
    while (j < fsz && count < (f_end - f_start)) {
      int fid = valid_fi[fids[j]];
      if (dense_f[fid]) {
        cudaMemcpyAsync(device, &(Xv[fid][0]), 
          sizeof(ushort)*n_data, cudaMemcpyHostToDevice, args[i].stream);
        device += n_data; 
        count++;
      }
      j++;
    }

    args[i].d_H = d_H;
    args[i].d_R = d_R;
    args[i].d_Xv = d_Xv;
    args[i].d_bc = d_bc;
    args[i].d_bs = d_bs;
		args[i].d_bw = d_bw;
    args[i].d_ids = d_ids;

    args[i].f_start = f_start;
    args[i].f_end = f_end;
    args[i].device_id = device_ids[i];
  } 

  for (int i = 0; i < n_available_devices; ++i) 
    cudaStreamSynchronize(args[i].stream);
}


/**
 * Copy hessians and residuals to GPU(s). 
 * @param[in] args: Vector of GPU arguments
 *            H: Hessian
 *            R: Residual 
 */
void updateGPUArgs(
  std::vector<GPUArgs> &args, std::vector<double> &H, std::vector<double> &R) {
  std::vector<float> H_f(H.begin(), H.end());
  std::vector<float> R_f(R.begin(), R.end());
  uint sz = H_f.size() * sizeof(float);
  for (int i = 0; i < args.size(); ++i) {
    cudaSetDevice(args[i].device_id);
    cudaMemcpyAsync(args[i].d_H, &(H_f[0]),
                    sz, cudaMemcpyHostToDevice, args[i].stream);
    cudaMemcpyAsync(args[i].d_R, &(R_f[0]),
                    sz, cudaMemcpyHostToDevice, args[i].stream);
  }

  for (int i = 0; i < args.size(); ++i)
    cudaStreamSynchronize(args[i].stream);
}


/**
 * Free all GPU memory.
 * @param[in] args: Vector of GPU arguments
 */
void freeDeviceMem(std::vector<GPUArgs> &args) {
  for (auto x : args) {
    cudaSetDevice(x.device_id);
    cudaFree(x.d_bc);
    cudaFree(x.d_bs);
    cudaFree(x.d_bw);
    cudaFree(x.d_H);
    cudaFree(x.d_R);
    cudaFree(x.d_ids);
    cudaFree(x.d_Xv);
  }
}

}
}  // namespace ABCBoost
