#pragma once
#include <vector>
#include <utility>
#include "mex.h"

std::vector<std::vector<std::pair<int,double>>> organize_sparse(const mxArray *x) {
  mxArray *prhs[1], *plhs[1];
  prhs[0] = mxDuplicateArray(x);
  mexCallMATLAB(1, plhs, 1, prhs, "transpose");
  mxArray* x_row = plhs[0];
  mxDestroyArray(prhs[0]);

  int n = mxGetN(x_row);
  double* vals = mxGetPr(x_row);
  auto* ir = mxGetIr(x_row);
  auto* jc = mxGetJc(x_row);

  std::vector<std::vector<std::pair<int,double>>> ret;
  for(int i = 0;i < n;++i){
    std::vector<std::pair<int,double>> row;
    for(int j = jc[i];j < jc[i + 1];++j){
      row.push_back(std::make_pair(ir[j] + 1, vals[j]));
    }
    ret.push_back(row);
  }
  return ret;
}
