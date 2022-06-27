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

#include <assert.h>
#include <math.h>
#include <algorithm>  // std::sort, std::min, std::max
#include <iterator>
#include <numeric>  // std::iota
#include <queue>    // std::priority_queue
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>

#include "config.h"
#ifdef OMP
#include "omp.h"
#else
#include "dummy_omp.h"
#endif
#include "tree.h"
#include "utils.h"


namespace ABCBoost {

inline HistBin csw_plus(const HistBin& a, const HistBin& b){
  return HistBin(a.count + b.count,a.sum + b.sum,a.weight + b.weight);
}

#pragma omp declare reduction(vec_csw_plus :  std::vector<HistBin>: \
  std::transform( \
    omp_out.begin(), omp_out.end(), \
    omp_in.begin(), omp_out.begin(), csw_plus)) \
  initializer(omp_priv = omp_orig)

#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
  std::transform( \
    omp_out.begin(), omp_out.end(), \
    omp_in.begin(), omp_out.begin(), std::plus<double>())) \
  initializer(omp_priv = omp_orig)

#pragma omp declare reduction(vec_int_plus: std::vector<int> : \
  std::transform( \
    omp_out.begin(), omp_out.end(), \
    omp_in.begin(), omp_out.begin(), std::plus<int>())) \
  initializer(omp_priv = omp_orig)

/**
 * Constructor.
 * @param[in] data: Dataset to train on
 *            config: Configuration
 */
Tree::Tree(Data *data, Config *config) {
  this->config = config;
  this->data = data;
  n_leaves = config->tree_max_n_leaves;
  n_threads = config->n_threads;
  nodes.resize(2 * n_leaves - 1);
  is_weighted = config->model_use_logit;
}

Tree::~Tree() {
  std::vector<short>().swap(leaf_ids);
  std::vector<TreeNode>().swap(nodes);
}

Tree::TreeNode::TreeNode() {
  is_leaf = true;
  idx = left = right = parent = -1;
  gain = predict_v = -1;
}


inline void Tree::alignHessianResidual(const uint start,const uint end){
  const auto* H = hessian;
  const auto* R = residual;
  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(static),
    config->use_omp == 1 && end - start > 1024,
    for(uint i = start;i < end;++i){
      auto id = ids[i];
      H_tmp[i] = H[id];
      R_tmp[i] = R[id];
    }
  )
}

inline void Tree::initUnobserved(const uint start,const uint end,double& r_unobserved, double& h_unobserved){
  const auto* H = hessian;
  const auto* R = residual;
  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(static) reduction(+: r_unobserved, h_unobserved),
    config->use_omp == true && (end - start > 1024),
    for (int i = start; i < end; ++i) {
      auto id = ids[i];
      r_unobserved += R[id];
      h_unobserved += H[id];
    }
  )
}


/**
 * Calculate bin_counts and bin_sums for all features at a node.
 * @param[in] x: Node id
 *            sib: Sibling id
 * @post bin_counts[x] and bin_sums[x] are populated.
 */
void Tree::binSort(int x, int sib) {
  const auto* H = hessian;
  const auto* R = residual;
  uint start = nodes[x].start;
  uint end = nodes[x].end;
  uint fsz = fids->size();

  if (sib == -1) {
    if(!(start == 0 && end == data->n_data)){
      alignHessianResidual(start,end);
    }

    double r_unobserved = 0.0;
    double h_unobserved = 0.0;
    int c_unobserved = end - start;
    initUnobserved(start,end,r_unobserved,h_unobserved);
    
    setInLeaf<true>(start,end);

    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(guided),
      config->use_omp == 1,
    for (int j = 0; j < fsz; ++j) {
      int fid = (data->valid_fi)[(*fids)[j]];
      auto &b_csw = (*hist)[x][fid];
      if(data->auxDataWidth[fid] == 0){
        std::vector<data_quantized_t> &fv = (data->Xv)[fid];
        if (data->dense_f[fid]) {
          if(start == 0 && end == data->n_data){
            for(uint i = start;i < end;++i){
              auto bin_id = fv[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[i];
              b_csw[bin_id].weight += is_weighted ? H[i] : 1;
            }
          }else{
            for(uint i = start;i < end;++i){
              auto bin_id = fv[ids[i]];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R_tmp[i];
              b_csw[bin_id].weight += is_weighted ? H_tmp[i] : 1;
            }
          }
        } else {
          std::vector<uint> &fi = (data->Xi)[fid];
          ushort j_unobserved = (data->data_header.unobserved_fv)[fid];

          // The following parallel for only works when fsz is very small
          // Remember to disable the outer parallel for before enable it, otherwise you will need to enable nested parallel for
          // #pragma omp parallel for schedule(static,1) reduction(vec_csw_plus:b_csw) if (fi.size() > 65536 * 8)
          for(int i = 0;i < fi.size();++i){
            if(in_leaf[fi[i]] == true){
              auto bin_id = fv[i];
              auto id = fi[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[id];
              b_csw[bin_id].weight += is_weighted ? H[id] : 1;
          
              b_csw[j_unobserved].sum -= R[id];
              b_csw[j_unobserved].count -= 1;
              b_csw[j_unobserved].weight -= is_weighted ? H[id] : 1;
            }
          }
          b_csw[j_unobserved].count += c_unobserved;
          b_csw[j_unobserved].sum += r_unobserved;
          b_csw[j_unobserved].weight += is_weighted ? h_unobserved : c_unobserved;
        }
      }else{
        std::vector<uint8_t> &fv = (data->auxData)[fid];
        if (data->dense_f[fid]) {
          if(start == 0 && end == data->n_data){
            for(uint i = start;i < end;++i){
              auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[i];
              b_csw[bin_id].weight += is_weighted ? H[i] : 1;
            }
          }else{
            for(uint i = start;i < end;++i){
              auto bin_id = (fv[ids[i] >> 1] >> ((ids[i] & 1) << 2)) & 15;
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R_tmp[i];
              b_csw[bin_id].weight += is_weighted ? H_tmp[i] : 1;
            }
          }
        } else {
          std::vector<uint> &fi = (data->Xi)[fid];
          ushort j_unobserved = (data->data_header.unobserved_fv)[fid];

          // The following parallel for only works when fsz is very small
          // Remember to disable the outer parallel for before enable it, otherwise you will need to enable nested parallel for
          // #pragma omp parallel for schedule(static,1) reduction(vec_csw_plus:b_csw) if (fi.size() > 65536 * 8)
          for(int i = 0;i < fi.size();++i){
            if(in_leaf[fi[i]] == true){
              auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
              auto id = fi[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[id];
              b_csw[bin_id].weight += is_weighted ? H[id] : 1;
          
              b_csw[j_unobserved].sum -= R[id];
              b_csw[j_unobserved].count -= 1;
              b_csw[j_unobserved].weight -= is_weighted ? H[id] : 1;
            }
          }
          b_csw[j_unobserved].count += c_unobserved;
          b_csw[j_unobserved].sum += r_unobserved;
          b_csw[j_unobserved].weight += is_weighted ? h_unobserved : c_unobserved;
        }
      
      }
    }
    )

    setInLeaf<false>(start,end);

  } else {
    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(guided),
      config->use_omp == true,
      for (int j = 0; j < fsz; ++j) {
        uint fid = (data->valid_fi)[(*fids)[j]];
        std::vector<HistBin> &b_csw = (*hist)[x][fid];
        int parent = nodes[x].parent;
        std::vector<HistBin> &pb_csw =
            (*hist)[parent][fid];
        std::vector<HistBin> &sb_csw = (*hist)[sib][fid];
        for (int k = 0; k < b_csw.size(); ++k) {
          b_csw[k].count = pb_csw[k].count - sb_csw[k].count;
          b_csw[k].sum = pb_csw[k].sum - sb_csw[k].sum;
          b_csw[k].weight =
              std::max((hist_t).0, pb_csw[k].weight - sb_csw[k].weight);
        }
      }
    )
  }
}

/**
 * Fit a decision tree to pseudo residuals which partitions the input space
 * into J disjoint regions and predicts a constant value for each region.
 * @param[in] ids: Pointer to sampled instance ids
 *            fids: Pointer to sampled feature ids
 * @post this->nodes and this->leaf_ids are populated.
 *       Feature importance is updated.
 */
void Tree::buildTree(std::vector<uint> *ids, std::vector<uint> *fids) {
  this->ids = (*(std::vector<uint> *)ids);
  this->fids = fids;
  in_leaf.resize(data->Y.size());

  dense_fids.reserve(fids->size());
  sparse_fids.reserve(fids->size());
  for(int j = 0;j < fids->size();++j){
    int fid = (data->valid_fi)[(*fids)[j]];
    if(data->dense_f[fid])
      dense_fids.push_back(fid);
    else
      sparse_fids.push_back(fid);
  }


  nodes[0].idx = 0;
  nodes[0].start = 0;
  nodes[0].end = ids->size();
  trySplit(0, -1);

  int l, r;
  uint lsz, rsz, msz = config->tree_min_node_size;

  const int n_iter = n_leaves - 1;
  for (int i = 0; i < n_iter; ++i) {
    // find the node with max gain to split (calculated in trySplit)
    int idx = -1;
    double max_gain = -1;
    for (int j = 0; j < 2 * i + 1; ++j) {
      if (nodes[j].is_leaf && nodes[j].gain > max_gain) {
        idx = j;
        max_gain = nodes[j].gain;
      }
    }
    l = 2 * i + 1;
    r = l + 1;
    if (idx == -1) {
      // TODO(@weijie): create dummy tree when above fails
      fprintf(stderr, "[INFO] cannot split further.\n");
      break;
    }
    split(idx, l);
    lsz = nodes[l].end - nodes[l].start, rsz = nodes[r].end - nodes[r].start;

    if (lsz < msz && rsz < msz) {
      fprintf(stderr,
              "[WARNING] Split is cancelled because of min node size!\n");
      continue;
    }
    
    if(i + 1 < n_iter){
      if (lsz < rsz) {
        trySplit(l, -1);
        //trySplit(r, -1); is replaced by the subtraction
        trySplit(r, l);
      } else {
        trySplit(r, -1);
        //trySplit(l, -1);
        trySplit(l, r);
      }
    }
  }
  regress();
  in_leaf.resize(0);
  in_leaf.shrink_to_fit();
}

void Tree::updateFeatureImportance(int iter) {
  for (double &x : (*feature_importance)) {
    x -= x / (iter + 1);
  }
  for (int i = 0; i < nodes.size(); ++i) {
    if (nodes[i].idx >= 0 && !nodes[i].is_leaf) {
      double tmp = nodes[i].gain / (iter + 1);
      if (tmp > 1e10) {
        tmp = 1e10;
      }
      (*feature_importance)[nodes[i].split_fi] += tmp;
    }
  }
}

/**
 * Compute the best split point for a feature at a node.
 * @param[in] x: Node id
 *            fid: Feature id
 */
std::pair<double, double> Tree::featureGain(int x, uint fid) const{
  auto &b_csw = (*hist)[x][fid];
  hist_t total_s = .0, total_w = .0;
  for (int i = 0; i < b_csw.size(); ++i) {
    total_s += b_csw[i].sum;
    total_w += b_csw[i].weight;
  }

  int l_c = 0, r_c = 0;
  hist_t l_w = 0, l_s = 0;
  int st = 0, ed = ((int)b_csw.size()) - 1;
  while (
      st <
      b_csw.size()) {  // st = min_i (\sum_{k <= i} counts[i]) >= min_node_size
    l_c += b_csw[st].count;
    l_s += b_csw[st].sum;
    l_w += b_csw[st].weight;
    if (l_c >= config->tree_min_node_size) break;
    ++st;
  }

  if (st == b_csw.size()) {
    return std::make_pair(-1, -1);
  }

  do {  // ed = max_i (\sum_{k > i} counts[i]) >= min_node_size
    r_c += b_csw[ed].count;
    ed--;
  } while (ed >= 0 && r_c < config->tree_min_node_size);

  if (st > ed) {
    return std::make_pair(-1, -1);
  }

  hist_t r_w = 0, r_s = 0;
  double max_gain = -1;
  int best_split_v = -1;
  for (int i = st; i <= ed; ++i) {
    if (b_csw[i].count == 0) {
      if (i + 1 < b_csw.size()) {
        l_w += b_csw[i + 1].weight;
        l_s += b_csw[i + 1].sum;
      }
      continue;
    }
    r_w = total_w - l_w;
    r_s = total_s - l_s;

    double gain = l_s / l_w * l_s + r_s / r_w * r_s;
    if (gain > max_gain /*&& gain < 1e10*/) {
      max_gain = gain;
      int offset = 1;
      while (i + offset < b_csw.size() && b_csw[i + offset].count == 0)
        offset++;
      best_split_v = i + offset / 2;
    }
    if (i + 1 < b_csw.size()) {
      l_w += b_csw[i + 1].weight;
      l_s += b_csw[i + 1].sum;
    }
  }

  max_gain -= total_s / total_w * total_s;
  return std::make_pair(max_gain, best_split_v);
}

/**
 * Clear ids to save memory.
 */
void Tree::freeMemory() {
  ids.clear();
  ids.shrink_to_fit();
}

/**
 * Assign pointers before building a tree (for testing).
 */
void Tree::init(std::vector<std::vector<uint>> *l_buffer,
                std::vector<std::vector<uint>> *r_buffer) {
  this->l_buffer = l_buffer;
  this->r_buffer = r_buffer;
  n_threads = config->n_threads;
}

/**
 * Assign pointers before building a tree (for training).
 */
void Tree::init(
    std::vector<std::vector<std::vector<HistBin>>>
        *hist,
    std::vector<std::vector<uint>> *l_buffer,
    std::vector<std::vector<uint>> *r_buffer,
    std::vector<double> *feature_importance, double *hessian,
    double *residual,
                uint* ids_tmp,
                double* H_tmp,
                double* R_tmp) {
  this->hist = hist;
  this->l_buffer = l_buffer;
  this->r_buffer = r_buffer;
  this->feature_importance = feature_importance;
  this->hessian = hessian;
  this->residual = residual;
  n_threads = config->n_threads;
  this->ids_tmp = ids_tmp;
  this->H_tmp = H_tmp;
  this->R_tmp = R_tmp;
}

/**
 * Load nodes for a saved tree.
 * @param[in] fileptr: Pointer to the FILE object
 *            n_nodes: Number of nodes
 */
void Tree::populateTree(FILE *fileptr) {
  int n_nodes = 0;
  size_t ret = fread(&n_nodes, sizeof(n_nodes), 1, fileptr);
  // use the actual tree size
  nodes.resize(n_nodes);

  int n_leafs = 0;
  int n = 0;
  for (n = 0; n < n_nodes; ++n) {
    TreeNode node = TreeNode();

    ret += fread(&node.idx, sizeof(node.idx), 1, fileptr);
    ret += fread(&node.parent, sizeof(node.parent), 1, fileptr);
    ret += fread(&node.left, sizeof(node.left), 1, fileptr);
    ret += fread(&node.right, sizeof(node.right), 1, fileptr);
    ret += fread(&node.split_fi, sizeof(node.split_fi), 1, fileptr);
    ret += fread(&node.split_v, sizeof(node.split_v), 1, fileptr);
    ret += fread(&node.predict_v, sizeof(node.predict_v), 1, fileptr);

    // check whether a leaf
    if (node.idx < 0) {
      node.is_leaf = false;
    } else if (node.left == -1 && node.right == -1) {
      n_leafs++;
      leaf_ids.push_back(node.idx);
    } else {
      node.is_leaf = false;
    }

    nodes[n] = node;
  }
}

/**
 * Predict region for a new instance.
 * @param[in] instance: All feature values of the instance
 * @return Region value
 */
double Tree::predict(std::vector<ushort> instance) {
  int i = 0;
  double predict_v;
  while (true) {
    // reach a leaf node
    if (nodes[i].is_leaf) {
      predict_v = nodes[i].predict_v;
      break;
    } else {  // go to left or right child
      i = instance[nodes[i].split_fi] <= nodes[i].split_v ? nodes[i].left
                                                          : nodes[i].right;
    }
  }
  return predict_v;
}

/**
 * Predict region for multiple instances.
 * @param[in] data: Dataset to train on
 * @return Region values for all instances.
 */
std::vector<double> Tree::predictAll(Data *data) {
  // use test data
  this->data = data;
  uint n_test = data->n_data;

  // initialize ids
  std::vector<uint> ids(n_test);
  std::iota(ids.begin(), ids.end(), 0);
  this->ids = ids;
  //TODO: fix this not init in test
  nodes[0].start = 0;
  nodes[0].end = n_test;

  std::vector<double> result(n_test, 0.0);

  for (int i = 0; i < nodes.size(); ++i) {
    // split at non-leaf
    if (nodes[i].idx < 0) continue;
    if (!nodes[i].is_leaf) split(i, nodes[i].left);
  }

  // instances now distributed in each leaf
  // return corresponding region value for each
  for (auto lfid : leaf_ids) {
    int start = nodes[lfid].start, end = nodes[lfid].end;
    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(static, 1),
      config->use_omp == true,
      for (int i = start; i < end; ++i)
        result[this->ids[i]] = nodes[lfid].predict_v;
    )
  }

  freeMemory();

  return result;
}

/**
 * Update region values for leaves.
 */
void Tree::regress() {
  double correction = 1.0;
  if (data->data_header.n_classes != 1 && config->model_name.size() >= 3 &&
      config->model_name.substr(0, 3) != "abc")
    correction -= 1.0 / data->data_header.n_classes;
  double upper = config->tree_clip_value, lower = -upper;

  auto* H = hessian;
  auto* R = residual;
  const bool is_weighted_update = config->model_use_weighted_update;

  for (int i = 0; i < nodes.size(); ++i) {
    if (nodes[i].idx >= 0 && nodes[i].is_leaf) {
      leaf_ids.push_back(i);
      double numerator = 0.0, denominator = 0.0;
      uint start = nodes[i].start, end = nodes[i].end;
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1) reduction(+: numerator, denominator),
        config->use_omp == true,
        for (uint d = start; d < end; ++d) {
          auto id = ids[d];
          numerator += R[id];
          denominator += H[id];
        }
      )
      nodes[i].predict_v =
          std::min(std::max(correction * numerator /
                                (denominator + config->tree_damping_factor),
                            lower),
                   upper);
    }
  }
}

/**
 * Save tree in a specified path.
 * @param[in] fp: Pointer to the FILE object
 */
void Tree::saveTree(FILE *fp) {
  int n = nodes.size();
  fwrite(&n, sizeof(n), 1, fp);
  for (TreeNode &node : nodes) {
    fwrite(&node.idx, sizeof(node.idx), 1, fp);
    fwrite(&node.parent, sizeof(node.parent), 1, fp);
    fwrite(&node.left, sizeof(node.left), 1, fp);
    fwrite(&node.right, sizeof(node.right), 1, fp);
    fwrite(&node.split_fi, sizeof(node.split_fi), 1, fp);
    fwrite(&node.split_v, sizeof(node.split_v), 1, fp);
    fwrite(&node.predict_v, sizeof(node.predict_v), 1, fp);
  }
}

/**
 * Partition instances at a node into its left and right child.
 * @param[in] x: Node id
 *            l: Left child id
 * @post Order of ids is updated.
 *       Start/end are stored for left (node[l]) and right child (node[l+1]).
 */
void Tree::split(int x, int l) {
  uint pstart = nodes[x].start;
  uint pend = nodes[x].end;
  uint n_ids = pend - pstart;

  int split_v = nodes[x].split_v;
  uint fid = nodes[x].split_fi, li = pstart, ri = 0;
  std::vector<data_quantized_t> &fv = (data->Xv)[fid];


  nodes[x].is_leaf = false;
  nodes[x].left = l;
  nodes[x].right = l + 1;
  nodes[l].idx = l;
  nodes[l].parent = x;
  nodes[l + 1].idx = l + 1;
  nodes[l + 1].parent = x;

  if ((data->dense_f)[fid]) {
    if (config->use_omp && n_ids > n_threads && n_threads > 1) {
      uint buffer_sz = (n_ids + n_threads - 1) / n_threads;
      std::vector<int> left_is(n_threads, 0), right_is(n_threads, 0);
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1) reduction(+ : li),
        config->use_omp == true,
        for (int t = 0; t < n_threads; ++t) {
          uint left_i = 0, right_i = 0;
          uint start = pstart + t * buffer_sz, end = start + buffer_sz;
          if (end > pend) end = pend;
          for (uint j = start; j < end; ++j) {
            uint id = ids[j];
            if (fv[id] <= split_v)
              (*l_buffer)[t][left_i++] = id;
            else
              (*r_buffer)[t][right_i++] = id;
          }

          left_is[t] = left_i;
          right_is[t] = right_i;
          li += left_i;
        }
      )

      std::vector<uint> left_sum(n_threads, 0), right_sum(n_threads, 0);
      for (int t = 1; t < n_threads; ++t) {
        left_sum[t] = left_sum[t - 1] + left_is[t - 1];
        right_sum[t] = right_sum[t - 1] + right_is[t - 1];
      }
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1),
        config->use_omp == true,
        for (int t = 0; t < n_threads; ++t) {
          std::move((*l_buffer)[t].begin(), (*l_buffer)[t].begin() + left_is[t],
                    ids.begin() + pstart + left_sum[t]);
          std::move((*r_buffer)[t].begin(), (*r_buffer)[t].begin() + right_is[t],
                    ids.begin() + li + right_sum[t]);
        }
      )
    } else {
      for (uint i = pstart; i < pend; ++i) {
        if(fv[ids[i]] <= split_v){
          ids[li] = ids[i];
          ++li;
        }else{
          ids_tmp[ri] = ids[i];
          ++ri;
        }
      }
      std::copy(ids_tmp, ids_tmp + ri, ids.begin() + li);
    }
  } else {
    int idx = 0;
    std::vector<uint> &fi = (data->Xi)[fid];
    int best_fsz = fi.size() - 1;
    ushort v, unobserved = (data->data_header.unobserved_fv)[fid];

    if (best_fsz >= 0) {
      for (uint i = pstart; i < pend; ++i) {
        uint id = ids[i];
        while (idx < best_fsz && fi[idx] < id) {
          ++idx;
        }
        v = (id == fi[idx]) ? fv[idx] : unobserved;
        if (v <= split_v)
          ids[li++] = id;
        else
          ids_tmp[ri++] = id;
      }
      std::copy(ids_tmp, ids_tmp + ri, ids.begin() + li);
    }
  }

  if (!(data->dense_f)[fid] && data->Xi[fid].size() <= 0) {
    li = ((data->data_header.unobserved_fv)[fid] <= split_v) ? pend : pstart;
  }
  nodes[l].start = pstart;
  nodes[l].end = li;
  nodes[l + 1].start = li;
  nodes[l + 1].end = pend;
}

/**
 * Compute the best feature to split as well as its information gain.
 * Meanwhile, store the bin sort results for later.
 * @param[in] x: Node id
 *            sib: Sibling id
 * @post gain, split_fi, and split_v are stored for node[x].
 */
void Tree::trySplit(int x, int sib) {
  binSort(x, sib);

  if ((nodes[x].end - nodes[x].start) < config->tree_min_node_size) return;
  SplitInfo best_info;

  best_info.gain = -1;
  std::vector<std::pair<double,int>> gains(fids->size());
  
  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(guided),
    config->use_omp == true,
    for (int j = 0; j < fids->size(); ++j) {
        int fid = (data->valid_fi)[(*fids)[j]];
        gains[j] = featureGain(x, fid);
    }
  )
  for(int j = 0;j < gains.size();++j){
    const auto& info = gains[j];
    int fid = (data->valid_fi)[(*fids)[j]];
    if (info.first > best_info.gain) {
      best_info.gain = info.first;
      best_info.split_fi = fid;
      best_info.split_v = info.second;
    }
  }

  if (best_info.gain < 0) return;
  nodes[x].gain = best_info.gain;
  nodes[x].split_fi = best_info.split_fi;
  nodes[x].split_v = best_info.split_v;
}

}  // namespace ABCBoost
