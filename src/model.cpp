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
#include <iomanip>

#include "config.h"
#include "data.h"
#include "model.h"
#include "tree.h"
#include "utils.h"

#ifdef OMP
#include "omp.h"
#else
#include "dummy_omp.h"
#endif

#if (defined(_WIN32) || defined(__WIN32__))
#define mkdir(A, B) mkdir(A)
#endif


namespace ABCBoost {

#ifndef OS_WIN
#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
  std::transform( \
    omp_out.begin(), omp_out.end(), \
    omp_in.begin(), omp_out.begin(), std::plus<double>())) \
  initializer(omp_priv = omp_orig)
#endif
// =============================================================================
//
// Gradient Boosting
//
// =============================================================================

/**
 * Constructor for the gradient boosting class - initialize and/or resizes
 * relevant variables defined in the model.h file for consistent and efficient
 * reference and indexing.
 * @param[in] data : pointer to data object as described in data.h, pointer
 *                   also stored as a field.
 *            config : pointer to config object as described in config.h,
 *                     pointer also stored as a field.
 * @return GradientBoosting object with populated fields.
 */
GradientBoosting::GradientBoosting(Data *data, Config *config) {
  this->data = data;
  this->config = config;
}

/**
 * Destructor for gradient boosting.
 */
GradientBoosting::~GradientBoosting() {
  if(log_out != NULL)
    fclose(log_out);
}

void GradientBoosting::print_test_message(int iter,double iter_time,int& low_err){
  if(config->no_label)
    return;
  double loss = getLoss();
  int err = getError();
  if(low_err > err)
    low_err = err;
  printf("%4d | loss: %20.14e | errors/lowest: %7d/%-7d | time: %.5f\n", iter,
       loss, err, low_err, iter_time);
#ifdef USE_R_CMD
 R_FlushConsole();
#endif
  if(config->save_log)
    fprintf(log_out,"%4d %20.14e %7d %.5f\n", iter, loss, err, iter_time);
}

void GradientBoosting::print_train_message(int iter,double loss,double iter_time){
  int err = getError();
  printf("%4d | loss: %20.14e | errors: %7d | time: %.5f\n", iter,
       loss, err, iter_time);
#ifdef USE_R_CMD
  R_FlushConsole();
#endif
  if(config->save_log)
    fprintf(log_out,"%4d %20.14e %7d %.5f\n", iter, loss, err, iter_time);
}

ModelHeader GradientBoosting::loadModelHeader(Config *config) {
  FILE *fp = fopen(config->model_pretrained_path.c_str(), "rb");
  if (config->model_pretrained_path == "" || fp == NULL) {
    ModelHeader ret = ModelHeader();
    ret.config.null_config = true;
    return ret;
  }
  ModelHeader model_header = ModelHeader::deserialize(fp);
  fclose(fp);
  return model_header;
}

void GradientBoosting::saveF() {
  FILE *fp = fopen(config->prediction_file.c_str(), "w");
  if (fp == NULL) {
    fprintf(stderr, "[Warning] prediction_file is not specified.\n");
    return;
  }
  for (size_t i = 0; i < data->n_data; ++i) {
    fprintf(fp, "%.5f\n", F[0][i]);
  }
  fclose(fp);
}

void GradientBoosting::returnPrediction(double *ret) {
  if (config->model_name == "lambdarank" || config->model_name == "lambdamart" || config->model_name == "gbrank" || config->model_name == "regression") {
    for (size_t i = 0; i < data->n_data; ++i) {
      ret[i] = F[0][i];
    }
  } else {
    for (size_t i = 0; i < data->n_data; ++i) {
      std::vector<double> prob(data->data_header.n_classes);
      for (int j = 0; j < data->data_header.n_classes; ++j) prob[j] = F[j][i];
      softmax(prob);
      for (int j = 0; j < data->data_header.n_classes; ++j) {
        int internal_idx = data->data_header.label2idx[j];
        ret[j * data->n_data + i] = prob[internal_idx];
      }
    }
  }
}

void GradientBoosting::savePrediction() {
  std::string prediction_file = config->formatted_output_name + ".prediction";
  std::string probability_file = config->formatted_output_name + ".probability";
  FILE *fp = fopen(prediction_file.c_str(), "w");
  FILE *fprob = NULL;
  if (config->model_name == "lambdarank" || config->model_name == "lambdamart" || config->model_name == "gbrank" || config->model_name == "regression") {
    for (size_t i = 0; i < data->n_data; ++i) {
      fprintf(fp, "%.5f ", F[0][i]);
      fprintf(fp, "\n");
    }
  }else {
    if (config->save_prob){
      fprob = fopen(probability_file.c_str(), "w");
    }
    for (size_t i = 0; i < data->n_data; ++i) {
      std::vector<double> prob(data->data_header.n_classes);
      double maxn = F[0][i];
      int maxj = 0;
      for (int j = 0; j < data->data_header.n_classes; ++j){
        prob[j] = F[j][i];
        if(maxn < prob[j]){
          maxn = prob[j];
          maxj = j;
        }
      }
      int pred = round(data->data_header.idx2label[maxj]);
      fprintf(fp,"%d\n",pred);
      if(fprob != NULL){
        softmax(prob);
        for (int j = 0; j < data->data_header.n_classes; ++j) {
          int internal_idx = data->data_header.label2idx[j];
          fprintf(fprob, "%.5f ", prob[internal_idx]);
        }
        fprintf(fprob, "\n");
      }
    }
  }
  fclose(fp);
  if(fprob != NULL)
    fclose(fprob);
}

/**
 * Test method.
 */
void GradientBoosting::test() { return; }

/**
 * Train method.
 */
void GradientBoosting::train() { return; }

/**
 * Method to get the argmax of a vector.
 * @param[in] vec: the vector
 * @return index of max value
 */
int GradientBoosting::argmax(std::vector<double> &vec) {
  int idx = 0;
  for (int j = 1; j < vec.size(); ++j) {
    if (vec[j] > vec[idx]) idx = j;
  }
  return idx;
}

/**
 * Method initializes and resizes data structures imperative in other methods
 */
void GradientBoosting::init() {
  int n_nodes = config->tree_max_n_leaves * 2 - 1;
  hist.resize(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    hist[i].resize(data->data_header.n_feats);
#pragma omp parallel for schedule(guided)
    for (unsigned int j = 0; j < data->data_header.n_feats; ++j) {
      hist[i][j].resize(data->data_header.n_bins_per_f[j]);
    }
  }

  F = std::vector<std::vector<double>>(data->data_header.n_classes,
                                       std::vector<double>(data->n_data, 0));
  hessians.resize(data->data_header.n_classes * data->n_data);
  residuals.resize(data->data_header.n_classes * data->n_data);

  //  additive_trees = std::vector<std::vector<std::unique_ptr<Tree>>>(
  //      config->model_n_iterations,
  //      std::vector<std::unique_ptr<Tree>>(data->data_header.n_classes));
  additive_trees.resize(config->model_n_iterations);
  for (int i = 0; i < additive_trees.size(); ++i)
    additive_trees[i].resize(data->data_header.n_classes);

  feature_importance.resize(data->data_header.n_feats, 0.0);
  
  R_tmp.resize(data->n_data);
  H_tmp.resize(data->n_data);
  ids_tmp.resize(data->n_data);
}

/**
 * Helper method to compute current accuracy on training data.
 * @return percentage accuracy over training set.
 */
double GradientBoosting::getAccuracy() {
  double accuracy = 0.0;
#pragma omp parallel for reduction(+ : accuracy)
  for (int i = 0; i < data->n_data; ++i) {
    int prediction = 0;
    for (int k = 1; k < data->data_header.n_classes; ++k)
      if (F[k][i] > F[prediction][i]) prediction = k;

    if (prediction == int(data->Y[i])) accuracy += 1;
  }
  return accuracy / data->n_data;
}

int GradientBoosting::getError() {
  int accuracy = 0;
#pragma omp parallel for reduction(+ : accuracy)
  for (int i = 0; i < data->n_data; ++i) {
    int prediction = 0;
    for (int k = 1; k < data->data_header.n_classes; ++k)
      if (F[k][i] > F[prediction][i])
        prediction = k;
    if (prediction == int(data->Y[i]))
      ++accuracy;
  }
  return data->n_data - accuracy;
}

/**
 * Helper method to compute CE loss on current probabilities.
 * @return summed CE loss over training set.
 */
double GradientBoosting::getLoss() {
  double loss = 0.0;
  if(data->data_header.n_classes == 2){
    #pragma omp parallel for reduction(+ : loss)
    for (int i = 0; i < data->n_data; i++) {
      if (data->Y[i] >= 0) {
        double curr = F[int(data->Y[i])][i];
        double tmp = -curr - curr;
        if (tmp > 700) tmp = 700;
        loss += log(1 + exp(tmp));
      }
    }
  }else{
    #pragma omp parallel for reduction(+ : loss)
    for (int i = 0; i < data->n_data; i++) {
      if (data->Y[i] >= 0) {
        double curr = F[int(data->Y[i])][i];
        double denominator = 0;
        for (int k = 0; k < data->data_header.n_classes; ++k) {
          double tmp = F[k][i] - curr;
          if (tmp > 700) tmp = 700;
          denominator += exp(tmp);
        }
        // get loss for one example and add it to the total
        loss += log(denominator);
      }
    }
  }
  return loss;
}

/**
 * Select features with the most cumulative gains.
 * @param[in] n: Number of top features to output.
 */
void GradientBoosting::getTopFeatures(int n) {
return;
  // initialize original index locations
  std::vector<unsigned int> idx(feature_importance.size());
  std::iota(idx.begin(), idx.end(), 0);
  if (n > idx.size()) n = idx.size();
  // sort indexes based on comparing values in feature_importance
  sort(idx.begin(), idx.end(), [&](unsigned int i1, unsigned int i2) {
    return feature_importance[i1] > feature_importance[i2];
  });

  printf("\nTop %d important features, id : cumulative gain\n", n);
  for (int i = 0; i < n; ++i) {
    printf("#%2d feature id: %5d | gain: %.8f\n", i + 1, idx[i],
           feature_importance[idx[i]]);
  }
}

std::string GradientBoosting::getDataName() {
  int data_name_start_nix = config->data_path.find_last_of('/') + 1;
  int data_name_start_win = config->data_path.find_last_of('\\') + 1;
  int data_name_start = std::max(data_name_start_nix, data_name_start_win);
  std::string data_name = config->data_path.substr(data_name_start);
  return data_name;
}

/**
 * Helper method to sample instances/features.
 * @param[in] n: total number of values from which to sample
 *            sample_rate: decimal indicating percentage of values to sample
 * @return vector containing indices of sampled values.
 */
std::vector<unsigned int> GradientBoosting::sample(int n, double sample_rate) {
  int n_samples = n * sample_rate;

  std::vector<unsigned int> indices(n_samples);
  std::iota(indices.begin(), indices.end(), 0);  // set values
  std::vector<bool> bool_indices(n, false);
  std::fill(bool_indices.begin(), bool_indices.begin() + n_samples, true);
  for (int i = n_samples + 1; i <= n; ++i) {
    int gen = (rand() % (i)) + 1;  // determine if need to swap
    if (gen <= n_samples) {
      int swap = rand() % n_samples;
      bool_indices[indices[swap]] = false;
      indices[swap] = i - 1;
      bool_indices[i - 1] = true;
    }
  }
  int index = 0;
  for (int i = 0; i < n; i++) {  // populate indices with sorted samples
    if (bool_indices[i]) {
      indices[index] = i;
      ++index;
    }
  }
  return indices;
}

/**
 * Save model.
 */
void GradientBoosting::saveModel(int iter) {
  FILE *model_out =
      fopen((experiment_path + config->model_suffix).c_str(), "wb");
  if (model_out == NULL) {
    fprintf(stderr, "[ERROR] Cannot create file: (%s)\n",
            (experiment_path + config->model_suffix).c_str());
    exit(1);
  }
  ModelHeader model_header;
  model_header.config = *config;
  model_header.config.model_n_iterations = iter;
  
  model_header.auxDataHeader = data->data_header;
  model_header.serialize(model_out);
  serializeTrees(model_out, iter);
  fclose(model_out);
  return;
}

/**
 * Helper method to setup files to save log information and the model.
 */
void GradientBoosting::setupExperiment() {
  int data_name_start_nix = config->data_path.find_last_of('/') + 1;
  int data_name_start_win = config->data_path.find_last_of('\\') + 1;
  int data_name_start = std::max(data_name_start_nix, data_name_start_win);
  std::string data_name = config->data_path.substr(data_name_start);

  struct stat buffer;
  if (stat(config->experiment_folder.c_str(), &buffer) != 0) {
#ifdef OS_WIN
    const int err = _mkdir(config->experiment_folder.c_str());
#else
    const int err = mkdir(config->experiment_folder.c_str(),
      S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
    if (err == -1) {
      fprintf(stderr, "[ERROR] Could not create experiment folder!\n");
      exit(1);
    }
  }
  std::ostringstream sstream;
  sstream << config->experiment_folder << data_name << "_" << config->model_name;
  if(config->model_name == "abcmart" || config->model_name == "abcrobustlogit"){
    sstream << config->base_candidates_size << "g" << config->model_gap;
    if(config->abc_sample_rate != 1)
      sstream << "s" << config->abc_sample_rate;
  }
  sstream << "_J" << config->tree_max_n_leaves << "_v"
          << config->model_shrinkage;
  if(config->model_name == "abcmart" || config->model_name == "abcrobustlogit"){
    sstream << "_w" << config->warmup_iter;
  }
  if(config->model_name == "regression"){
    sstream << "_p" << config->regression_lp_loss;
  }

  experiment_path = sstream.str();
  config->formatted_output_name = experiment_path;

  log_out = (config->save_log && config->no_label == false) ? fopen((experiment_path + "." + config->model_mode + "log").c_str(), "w") : stdout;

  sample_data = (config->model_data_sample_rate < 1);
  sample_feature = (config->model_feature_sample_rate < 1);

  ids.resize(data->n_data);
  fids.resize(data->valid_fi.size());
  if (!sample_data) std::iota(ids.begin(), ids.end(), 0);
  if (!sample_feature) std::iota(fids.begin(), fids.end(), 0);

  printf(
      "\nModel Summary: | model: %s | mode: %s |"
      " max # leaves: %d | shrinkage: %4.2f |\n\n",
      config->model_name.c_str(), config->model_mode.c_str(),
      config->tree_max_n_leaves, config->model_shrinkage);
}

/**
 * Helper method to perform stable row-softmax.
 * Subtract max value of each row from other elements for stability.
 * @param[in] v : current training example to perform softmax over.
 */
void GradientBoosting::softmax(std::vector<double> &v) {
  double max = v[0], normalization = 0;
  int j, sz = v.size();

  auto v2 = v;
  // find max value
  for (j = 1; j < sz; ++j) {
    max = std::max(max, v[j]);
  }

  for (j = 0; j < sz; ++j) {
    double tmp = v[j] - max;
    if (tmp > 700) tmp = 700;
    v[j] = exp(tmp);
    normalization += v[j];
  }

  // normalize
  for (j = 0; j < sz; ++j) {
    v[j] /= normalization;
  }
}

/**
 * Helper method to update Fvalues based on fitted regression tree.
 * @param[in] k:    current class for which f-value matrix is being updated.
 *            tree: pointer to fitted regression tree, for access to
 *                  incremental updates.
 */
void GradientBoosting::updateF(int k, Tree *tree) {
  std::vector<unsigned int> &ids = tree->ids;
  std::vector<double> &f = F[k];
  for (auto leaf_id : tree->leaf_ids) {
    if (leaf_id < 0) {
      // printf("found negative leaf id\n");
      continue;
    }
    const Tree::TreeNode& node = tree->nodes[leaf_id];
    double update = config->model_shrinkage * node.predict_v;
    unsigned int start = node.start, end = node.end;
#pragma omp parallel for
    for (int i = start; i < end; ++i) f[ids[i]] += update;
  }
  tree->freeMemory();
}

void ABCMart::updateNormF(int k, Tree *tree) {
  std::vector<unsigned int> &ids = tree->ids;
  std::vector<double> &f = F[k];
  std::vector<double> norm_f(f.size(),0);

  for (auto leaf_id : tree->leaf_ids) {
    if (leaf_id < 0) {
      // printf("found negative leaf id\n");
      continue;
    }
    const Tree::TreeNode& node = tree->nodes[leaf_id];
    double update = config->model_shrinkage * node.predict_v;
    unsigned int start = node.start, end = node.end;
#pragma omp parallel for
    for (int i = start; i < end; ++i) norm_f[ids[i]] += update;
  }
  double sum = 0;
  for(int i = 0;i < norm_f.size();++i){
    sum += norm_f[i];
  }
  sum /= 2;
  for(int i = 0;i < norm_f.size();++i){
    f[i] += norm_f[i] - sum;
  }
  tree->freeMemory();
}

/**
 * Helper method to zero out bin counts and bin sums.
 */
void GradientBoosting::zeroBins() {
  int n_nodes = config->tree_max_n_leaves * 2 - 1;
  int n_feats = data->data_header.n_feats;
#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < n_nodes; ++i) {
    for (int j = 0; j < n_feats; ++j) {
      memset(hist[i][j].data(), 0, sizeof(HistBin) * hist[i][j].size());
    }
  }
}

// =============================================================================
//
// Regression
//
// =============================================================================

/**
 * Mart Constructor.
 * @param[in] data: pointer to Data object as required by GradientBoosting
 *                  constructor
 *            config: pointer to Config object as required by GradientBoosting
 *                    constructor
 * @return Mart object containing properties of Mart model.
 */

Regression::Regression(Data *data, Config *config)
    : GradientBoosting(data, config) {}

void Regression::init(){
  GradientBoosting::init();
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
    if(data->Y.size() > 0){
      double prev = config->regression_stop_factor;
      config->regression_stop_factor = pow(config->regression_stop_factor, p / 2.0) * sump / data->n_data;
      //printf("[INFO] regression_stop factor changed to pow(%f, %f) * %f / %d = %f\n",prev, p / 2.0,sump,data->n_data,config->regression_stop_factor);
    }
    if(config->regression_auto_clip_value && maxn - minn > 0){
      config->tree_clip_value *= maxn - minn;
      //printf("[INFO] automatically use %f as tree_clip_value based on y range [%f, %f]\n",config->tree_clip_value,minn,maxn);
    }
  }
}

void Regression::print_train_message(int iter,double loss,double iter_time){
  printf("%4d | loss: %20.14e | time: %.5f\n", iter,
       loss, iter_time);
#ifdef USE_R_CMD
  R_FlushConsole();
#endif
  if(config->save_log)
    fprintf(log_out,"%4d %20.14e %.5f\n", iter, loss, iter_time);
}

void Regression::print_test_message(int iter,double iter_time,double& low_loss){
  if(config->no_label)
    return;
  double loss = config->regression_huber_loss ? getHuberLoss() : getLpLoss();
  std::string loss_name = "";
  if(config->regression_huber_loss)
    loss_name = "huber_loss";
  else if(config->regression_lp_loss == 1.0)
    loss_name = "l1_loss";
  else if(config->regression_lp_loss != 2.0)
    loss_name = "l" + std::to_string(config->regression_lp_loss) + "_loss";
  else
    loss_name = "l2_loss";

  if(low_loss > loss)
    low_loss = loss;
  double l2_loss = loss;
  if(loss_name != "l2_loss"){
    l2_loss = getLSLoss();
    printf("%4d | l2_loss: %20.14e | %s: %-20.14e | time: %.5f\n", iter,
       l2_loss, loss_name.c_str(), loss, iter_time);
  }else{
    printf("%4d | l2_loss: %20.14e | time: %.5f\n", iter,
       loss, iter_time);
  }
    
#ifdef USE_R_CMD
 R_FlushConsole();
#endif
  if(config->save_log){
    if(loss_name != "l2_loss"){
      fprintf(log_out,"%4d %20.14e %20.14e %.5f\n", iter, l2_loss, loss, iter_time);
    }else{
      fprintf(log_out,"%4d %20.14e %.5f\n", iter, loss, iter_time);
    }
  }
}

/**
 * Method to implement testing process for MART algorithm as described by
 * Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 */
void Regression::test() {
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
      /*      data->loadRankQuery();
            auto p = getNDCG();
            printf("Time: %f | NDCG0 %f NDCG1 %f\n",
              t1.get_time_restart(), p.first, p.second);*/
    } else {
      if ((m + 1) % config->model_eval_every == 0){
        print_test_message(m + 1,t1.get_time_restart(),low_err);
      }
    }
  }
}

/**
 * Method to implement training process for MART algorithm as described
 * by Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 * @param[in] start: index to start training at.
 */
void Regression::train() {
  // set up buffers for OpenMP
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  // build one tree if it is binary prediction
  int K = 1;

  Utils::Timer t1, t2, t3;
  t1.restart();
  t2.restart();
  t3.restart();
  for (int m = start_epoch; m < config->model_n_iterations; m++) {
    if (config->model_data_sample_rate < 1)
      ids = sample(data->n_data, config->model_data_sample_rate);
    if (config->model_feature_sample_rate < 1)
      fids =
          sample(data->data_header.n_feats, config->model_feature_sample_rate);

    computeHessianResidual();

    zeroBins();

    Tree *tree;
    tree = new Tree(data, config);
    tree->init(&hist, &buffer[0], &buffer[1], &feature_importance,
               &(hessians[0]), &(residuals[0]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
    tree->buildTree(&ids, &fids);
    tree->updateFeatureImportance(m);
    updateF(0, tree);
    additive_trees[m][0] = std::unique_ptr<Tree>(tree);
    auto loss = config->regression_huber_loss ? getHuberLoss() : getLpLoss();
    if ((m + 1) % config->model_eval_every == 0)
      print_train_message(m + 1,loss,t1.get_time_restart());
    if (config->save_model && (m + 1) % config->model_save_every == 0) saveModel(m + 1);
    if(loss < config->regression_stop_factor){
        config->model_n_iterations = m + 1;
        break;
    }
  }
  printf("Training has taken %.5f seconds\n", t2.get_time());

  if (config->save_model) saveModel(config->model_n_iterations);
  getTopFeatures();

}

/**
 * Helper method to compute hessian and residual simultaneously.
 * @param k : the current class
 */
void Regression::computeHessianResidual() {
  if (config->regression_l1_loss){
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < data->n_data; i++) {
      residuals[i] = (F[0][i] - data->Y[i]) > 0 ? -1.0 : 1.0;
      hessians[i] = 1;
    }
  }else if (config->regression_huber_loss){
    const double& p = config->regression_lp_loss;
    const double delta = config->huber_delta;
  const double delta_p_1 = pow(delta,p - 1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < data->n_data; i++) {
      if(fabs(F[0][i] - data->Y[i]) <= delta){
        residuals[i] = -p * pow(fabs(F[0][i] - data->Y[i]),p - 1);
        hessians[i] = 1;
      }else{
        residuals[i] = ((F[0][i] - data->Y[i]) > 0 ? -1.0 : 1.0) * delta_p_1;
        hessians[i] = 1;
      }
    }
  }else if (config->regression_lp_loss != 2.0){
    const double& p = config->regression_lp_loss;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < data->n_data; i++) {
      int sign = (F[0][i] - data->Y[i]) > 0 ? -1 : 1;
      const double diff = fabs(F[0][i] - data->Y[i]);
      residuals[i] = p * pow(diff,p - 1) * sign;
      hessians[i] = (p <= 2) ? p : p * (p - 1) * pow(diff,p - 2);
    }
  }else{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < data->n_data; i++) {
      residuals[i] = -2.0 * (F[0][i] - data->Y[i]);
      hessians[i] = 2;
    }
  }
}

/**
 * Helper method to compute least squares loss on current probabilities.
 * @return summed least squares loss over training set.
 */
double Regression::getLSLoss() {
  double loss = 0.0;
  for (int i = 0; i < data->n_data; i++) {
    loss += (F[0][i] - data->Y[i]) * (F[0][i] - data->Y[i]);
  }
  return loss / data->n_data;
}

double Regression::getL1Loss() {
  double loss = 0.0;
  for (int i = 0; i < data->n_data; i++) {
    loss += fabs(F[0][i] - data->Y[i]);
  }
  return loss / data->n_data;
}

double Regression::getHuberLoss() {
  double loss = 0.0;
  const double& p = config->regression_lp_loss;
  const double delta = config->huber_delta;
  const double delta_p_1 = pow(delta,p - 1);
  for (int i = 0; i < data->n_data; i++) {
    if(fabs(F[0][i] - data->Y[i]) <= delta)
      loss += pow(fabs(F[0][i] - data->Y[i]),p);
    else
      loss += delta_p_1 * (2 * fabs(F[0][i] - data->Y[i]) - delta);
  }
  return loss / data->n_data;
}

double Regression::getLpLoss() {
  const double& p = config->regression_lp_loss;
  if(p == 1)
    return getL1Loss();
  if(p == 2)
    return getLSLoss();
  double loss = 0.0;
  for (int i = 0; i < data->n_data; i++) {
    loss += pow(fabs(F[0][i] - data->Y[i]),p);  
  }
  return loss / data->n_data;
}

/**
 * Helper method to load the pre-trained model.
 * @return final training iteration of loaded model.
 */
int Regression::loadModel() { return GradientBoosting::loadModel(); }

/**
 * Helper method to save the current model.
 */
void Regression::saveModel(int iter) { GradientBoosting::saveModel(iter); }




BinaryMart::BinaryMart(Data *data, Config *config) : GradientBoosting(data, config) {}

void BinaryMart::savePrediction() {
  std::string prediction_file = config->formatted_output_name + ".prediction";
  std::string probability_file = config->formatted_output_name + ".probability";
  FILE *fp = fopen(prediction_file.c_str(), "w");
  FILE *fprob = NULL;
  if (config->save_prob){
    fprob = fopen(probability_file.c_str(), "w");
  }
  for (size_t i = 0; i < data->n_data; ++i) {
    std::vector<double> prob(2);
    prob[0] = F[i];
    prob[1] = -F[i];
    int maxj = prob[0] >= prob[1] ? 0 : 1;
    int pred = round(data->data_header.idx2label[maxj]);
    fprintf(fp,"%d\n",pred);
    if(fprob != NULL){
      softmax(prob);
      for (int j = 0; j < data->data_header.n_classes; ++j) {
        int internal_idx = data->data_header.label2idx[j];
        fprintf(fprob, "%.5f ", prob[internal_idx]);
      }
      fprintf(fprob, "\n");
    }
  }
  fclose(fp);
  if(fprob != NULL)
    fclose(fprob);
}

void BinaryMart::returnPrediction(double* ret) {
  for (size_t i = 0; i < data->n_data; ++i) {
    std::vector<double> prob(2);
    prob[0] = F[i];
    prob[1] = -F[i];
    int maxj = prob[0] >= prob[1] ? 0 : 1;
    int pred = round(data->data_header.idx2label[maxj]);
    softmax(prob);
    for (int j = 0; j < data->data_header.n_classes; ++j) {
      int internal_idx = data->data_header.label2idx[j];
      ret[j * data->n_data + i] = prob[internal_idx];
    }
  }
}

void BinaryMart::test() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  Utils::Timer t1;
  t1.restart();

  double best_accuracy = 0;
  int low_err = std::numeric_limits<int>::max();
  for (int m = 0; m < config->model_n_iterations; ++m) {
    additive_trees[m][0]->init(nullptr, &buffer[0], &buffer[1], nullptr,
                               nullptr, nullptr,ids_tmp.data(),H_tmp.data(),R_tmp.data());
    std::vector<double> updates = additive_trees[m][0]->predictAll(data);
    for (int i = 0; i < data->n_data; ++i) {
      F[i] += config->model_shrinkage * updates[i];
    }


    if ((m + 1) % config->model_eval_every == 0){
      print_test_message(m + 1,t1.get_time_restart(),low_err);
    }
  }
}

void BinaryMart::updateF(Tree *tree) {
  std::vector<unsigned int> &ids = tree->ids;
  for (auto leaf_id : tree->leaf_ids) {
    if (leaf_id < 0) {
      // printf("found negative leaf id\n");
      continue;
    }
    const Tree::TreeNode& node = tree->nodes[leaf_id];
    double update = config->model_shrinkage * node.predict_v;
    unsigned int start = node.start, end = node.end;
#pragma omp parallel for
    for (int i = start; i < end; ++i) F[ids[i]] += update;
  }
  tree->freeMemory();
}
void BinaryMart::computeHessianResidual() {
#pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < data->n_data; ++i) {
    int label = int(data->Y[i]);
    double tmp = -F[i] - F[i];
    if(tmp > 700)
      tmp = 700;
    double p_i = 1.0 / (1 + exp(tmp));
    residuals[i] = (0 == label) ? (1 - p_i) : -p_i;
    hessians[i] = p_i * (1 - p_i);
  }
}

void BinaryMart::init() {
  GradientBoosting::init();
  F.resize(data->n_data);
}
double BinaryMart::getAccuracy() {
  double accuracy = 0.0;
#pragma omp parallel for reduction(+ : accuracy)
  for (int i = 0; i < data->n_data; ++i) {
    int label = int(data->Y[i]);
    if ((label == 0) == (F[i] >= 0))
      accuracy += 1;
  }
  return accuracy / data->n_data;
}

int BinaryMart::getError() {
  int accuracy = 0;
#pragma omp parallel for reduction(+ : accuracy)
  for (int i = 0; i < data->n_data; ++i) {
    int label = int(data->Y[i]);
    if ((label == 0) == (F[i] >= 0))
      ++accuracy;
  }
  return data->n_data - accuracy;
}

double BinaryMart::getLoss() {
  double loss = 0.0;
  #pragma omp parallel for reduction(+ : loss)
  for (int i = 0; i < data->n_data; i++) {
    int label = data->Y[i];
    double tmp = (label == 0) ? -F[i] - F[i] : F[i] + F[i];
    if (tmp > 700) tmp = 700;
    loss += log(1 + exp(tmp));
  }
  return loss;
}

void BinaryMart::train() {
  // set up buffers for OpenMP
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  // build one tree if it is binary prediction

  Utils::Timer t1, t2, t3;
  t1.restart();
  t2.restart();
  t3.restart();

  for (int m = start_epoch; m < config->model_n_iterations; m++) {
    if (config->model_data_sample_rate < 1)
      ids = sample(data->n_data, config->model_data_sample_rate);
    if (config->model_feature_sample_rate < 1)
      fids =
          sample(data->data_header.n_feats, config->model_feature_sample_rate);
    computeHessianResidual();

    zeroBins();
    Tree *tree;
    tree = new Tree(data, config);
    tree->init(&hist, &buffer[0], &buffer[1], &feature_importance,
               &(hessians[0]), &(residuals[0]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
    tree->buildTree(&ids, &fids);
    tree->updateFeatureImportance(m);
    updateF(tree);
    additive_trees[m][0] = std::unique_ptr<Tree>(tree);

    double loss = getLoss();
    if ((m + 1) % config->model_eval_every == 0){
      print_train_message(m + 1,loss,t1.get_time_restart());
    }
    if (config->save_model && (m + 1) % config->model_save_every == 0) saveModel(m + 1);
    if(loss < config->stop_tolerance){
      config->model_n_iterations = m + 1;
      break;
    }
  }
  printf("Training has taken %.5f seconds\n", t2.get_time());

  if (config->save_model) saveModel(config->model_n_iterations);

  getTopFeatures();
}


// =============================================================================
//
// Mart
//
// =============================================================================

/**
 * Mart Constructor.
 * @param[in] data: pointer to Data object as required by GradientBoosting
 *                  constructor
 *            config: pointer to Config object as required by GradientBoosting
 *                    constructor
 * @return Mart object containing properties of Mart model.
 */
Mart::Mart(Data *data, Config *config) : GradientBoosting(data, config) {}

/**
 * Method to implement testing process for MART algorithm as described by
 * Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 */
void Mart::test() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  Utils::Timer t1;
  t1.restart();

  double best_accuracy = 0;
  int K = (data->data_header.n_classes == 2) ? 1 : data->data_header.n_classes;
  int low_err = std::numeric_limits<int>::max();
  for (int m = 0; m < config->model_n_iterations; ++m) {
    for (int k = 0; k < K; ++k) {
      if (additive_trees[m][k] != NULL) {
        additive_trees[m][k]->init(nullptr, &buffer[0], &buffer[1], nullptr,
                                   nullptr, nullptr,ids_tmp.data(),H_tmp.data(),R_tmp.data());
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

    if ((m + 1) % config->model_eval_every == 0){
      print_test_message(m + 1,t1.get_time_restart(),low_err);
    }
  }
}

std::vector<std::vector<std::vector<unsigned int>>>
GradientBoosting::initBuffer() {
  std::vector<std::vector<std::vector<unsigned int>>> ret(2);
  int n_threads = 1;
#pragma omp parallel
#pragma omp master
  {
    n_threads = config->n_threads;  // omp_get_num_threads();
  }
  omp_set_num_threads(n_threads);
  config->n_threads = n_threads;
  unsigned int buffer_sz = (data->n_data + n_threads - 1) / n_threads;
  ret[0] = ret[1] = std::vector<std::vector<unsigned int>>(
      n_threads, std::vector<unsigned int>(buffer_sz, 0));
  return ret;
}

/**
 * Method to implement training process for MART algorithm as described
 * by Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 * @param[in] start: index to start training at.
 */
void Mart::train() {
  // set up buffers for OpenMP
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  // build one tree if it is binary prediction
  int K = (data->data_header.n_classes == 2) ? 1 : data->data_header.n_classes;

  Utils::Timer t1, t2, t3;
  t1.restart();
  t2.restart();
  t3.restart();

  for (int m = start_epoch; m < config->model_n_iterations; m++) {
    if (config->model_data_sample_rate < 1)
      ids = sample(data->n_data, config->model_data_sample_rate);
    if (config->model_feature_sample_rate < 1)
      fids =
          sample(data->data_header.n_feats, config->model_feature_sample_rate);
    
    computeHessianResidual();

    for (int k = 0; k < K; ++k) {
      
      zeroBins();
      Tree *tree;
      tree = new Tree(data, config);
      tree->init(&hist, &buffer[0], &buffer[1], &feature_importance,
                 &(hessians[k * data->n_data]), &(residuals[k * data->n_data]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
      tree->buildTree(&ids, &fids);
      tree->updateFeatureImportance(m);
      updateF(k, tree);
      additive_trees[m][k] = std::unique_ptr<Tree>(tree);
    }
    if (data->data_header.n_classes == 2) {
#pragma omp parallel for
      for (int i = 0; i < data->n_data; ++i) F[1][i] = -F[0][i];
    }

    double loss = getLoss();
    if ((m + 1) % config->model_eval_every == 0){
      print_train_message(m + 1,loss,t1.get_time_restart());
    }
    if (config->save_model && (m + 1) % config->model_save_every == 0) saveModel(m + 1);
    if(loss < config->stop_tolerance){
      config->model_n_iterations = m + 1;
      break;
    }
  }
  printf("Training has taken %.5f seconds\n", t2.get_time());

  if (config->save_model) saveModel(config->model_n_iterations);

  getTopFeatures();

}

/**
 * Helper method to compute hessian and residual simultaneously.
 */
void Mart::computeHessianResidual() {
  std::vector<double> prob;
#pragma omp parallel for schedule(static) private(prob)
  for (unsigned int i = 0; i < data->n_data; ++i) {
    prob.resize(data->data_header.n_classes);
    int label = int(data->Y[i]);
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      prob[k] = F[k][i];
    }
    softmax(prob);
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      double p_ik = prob[k];
      residuals[k * data->n_data + i] = (k == label) ? (1 - p_ik) : -p_ik;
      hessians[k * data->n_data + i] = p_ik * (1 - p_ik);
    }
  }
}

/**
 * Helper method to load the pre-trained model.
 * @return final training iteration of loaded model.
 */
int GradientBoosting::loadModel() {
  FILE *fp = fopen(config->model_pretrained_path.c_str(), "rb");
  if (fp == NULL) return -1;
  // retrieve trees
  ModelHeader model_header = ModelHeader::deserialize(fp);
  GradientBoosting::deserializeTrees(fp);
  fclose(fp);
  return 0;
}

void GradientBoosting::serializeTrees(FILE *fp, int M) {
  int K = M > 0 ? additive_trees[0].size() : 0;
  Utils::serialize(fp, M);
  Utils::serialize(fp, K);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < K; ++j) {
      if (additive_trees[i][j] == nullptr) {
        int n = 0;
        fwrite(&n, sizeof(n), 1, fp);
        continue;
      }
      additive_trees[i][j]->saveTree(fp);
    }
}

void GradientBoosting::deserializeTrees(FILE *fp) {
  int M = Utils::deserialize<int>(fp);
  int K = Utils::deserialize<int>(fp);

  int N = 2 * config->tree_max_n_leaves - 1;  // number of nodes

  if(config->model_mode == "test" && M < config->model_n_iterations){
    fprintf(stderr,"[Warning] Command line specifies %d iterations, while the model is only trained with %d iterations!\n",config->model_n_iterations,M);
    config->model_n_iterations = M;
  }
  if(config->model_mode == "train" && M >= config->model_n_iterations){
    fprintf(stderr,"[Warning] Command line specifies %d iterations, while the model has already been trained with %d iterations! No need to do anyting.\n",config->model_n_iterations,M);
    exit(0);
  }

  for (int i = 0; i < M; ++i){
    for (int j = 0; j < K; ++j) {
      if(i >= config->model_n_iterations){
        auto dummy_tree = std::unique_ptr<Tree>(new Tree(data, config));
        dummy_tree->populateTree(fp);
        continue;
      }
      additive_trees[i][j] = std::unique_ptr<Tree>(new Tree(data, config));
      additive_trees[i][j]->populateTree(fp);
    }
  }
}

// =============================================================================
//
// ABCMart
//
// =============================================================================

/**
 * ABCMart Constructor.
 * Adds base_classes and class_losses fields in order to account for
 * storage of base classes and information to help adaptive base class
 * selection.
 * @param[in] data: pointer to Data object as required by GradientBoosting
 *            config: pointer to Config object as required by GradientBoosting
 * @return ABCMart model instance with populated properties.
 */
ABCMart::ABCMart(Data *data, Config *config) : GradientBoosting(data, config) {}

/**
 * Method to implement training process for ABCMART algorithm as described
 * by Ping Li (2008). Descriptions for used sub-routines are available
 * in the individual method-comments.
 * @param[in] start: index to start training at.
 */
void ABCMart::train() {
  int n_skip = 1;
  if(config->model_gap != "")
    n_skip = atoi(config->model_gap.c_str()) + 1;
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  bool sample_data = (config->model_data_sample_rate < 1);
  bool sample_feature = (config->model_feature_sample_rate < 1);
  int K = data->data_header.n_classes;

  Utils::Timer t1, t2;
  t1.restart();
  t2.restart();
  std::vector<int> base_candidates;
  if(config->base_candidates_size == 0){
    for(int i = 0;i < data->data_header.n_classes;++i)
      base_candidates.push_back(i);
  }
  for (int m = start_epoch; m < config->model_n_iterations; m++) {
    if (config->model_data_sample_rate < 1)
      ids = sample(data->n_data, config->model_data_sample_rate);
    if (config->model_feature_sample_rate < 1)
      fids =
          sample(data->data_header.n_feats, config->model_feature_sample_rate);

    if(m < config->warmup_iter){
      computeHessianResidual();
      for (int k = 0; k < K; ++k) {
        zeroBins();
        Tree *tree;
        tree = new Tree(data, config);
        tree->init(&hist, &buffer[0], &buffer[1], &feature_importance,
                   &(hessians[k * data->n_data]), &(residuals[k * data->n_data]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
        tree->buildTree(&ids, &fids);
        updateF(k, tree);
        additive_trees[m][k] = std::unique_ptr<Tree>(tree);
      }
    }else if ((m - config->warmup_iter) % n_skip == 0) {
      if (config->warmup_iter > 0 && m == config->warmup_iter){
        #pragma omp parallel for schedule(static)
        for (int i = 0;i < data->n_data;++i){
          double sum = 0;
          for (int k = 0;k < K;++k){
            sum += F[k][i];
          }
          sum /= K;
          for (int k = 0;k < K;++k){
            F[k][i] -= sum;
          }
        }
      }

      double best_base_loss = std::numeric_limits<double>::max();
      double curr_base_loss;
      std::vector<std::unique_ptr<Tree>> best_trees(K);
      std::vector<std::vector<double>> f_values_copy = F;
      std::vector<std::vector<double>> best_F;
      if(config->base_candidates_size != 0){
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

      std::vector<unsigned int> abc_sample_ids;
      bool abc_sample_mode = false;
      if(config->abc_sample_rate != 1 && base_candidates.size() > 1){
        abc_sample_mode = true;
        double rate = config->abc_sample_rate;
        if(rate * data->n_data < config->abc_sample_min_data)
          rate = config->abc_sample_min_data / data->n_data;
        abc_sample_ids = sample(data->n_data, config->abc_sample_rate);
      }
      for(int b : base_candidates){
        // printf("trying base class %d %d\n",b,(int)data->data_header.idx2label[b]);
        if(abc_sample_mode){
          computeHessianResidual(b,abc_sample_ids);
        }else{
          computeHessianResidual(b);
        }
        for (int k = 0; k < K; ++k) {
          if (k == b) continue;
          //printf("building class %d\n",(int)data->data_header.idx2label[k]);
          zeroBins();
          Tree *tree;
          tree = new Tree(data, config);
          tree->init(&hist, &buffer[0], &buffer[1], &feature_importance,
                     &(hessians[k * data->n_data]), &(residuals[k * data->n_data]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
          if(abc_sample_mode){
            tree->buildTree(&abc_sample_ids, &fids);
          }else{
            tree->buildTree(&ids, &fids);
          }
          updateF(k, tree);
          additive_trees[m][k] = std::unique_ptr<Tree>(tree);
        }

        if(abc_sample_mode){
          curr_base_loss = getLossRaw(b,abc_sample_ids);
          if (curr_base_loss < best_base_loss) {
            best_base_loss = curr_base_loss;
            base_classes[m] = b;
          }
        }else{
          updateBaseFValues(b);
          curr_base_loss = getLoss();
          if (curr_base_loss < best_base_loss) {
            best_base_loss = curr_base_loss;
            base_classes[m] = b;
            if(abc_sample_mode == false){
              best_trees.swap(additive_trees[m]);
              best_F = F;
            }
          }
        }
        F = f_values_copy;
      }
      if(abc_sample_mode){
        int b = base_classes[m];
        computeHessianResidual(b);
        for (int k = 0; k < K; ++k) {
          if (k == b) continue;
          zeroBins();
          Tree *tree;
          tree = new Tree(data, config);
          tree->init(&hist, &buffer[0], &buffer[1], &feature_importance,
                     &(hessians[k * data->n_data]), &(residuals[k * data->n_data]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
          tree->buildTree(&ids, &fids);
          updateF(k, tree);
          additive_trees[m][k] = std::unique_ptr<Tree>(tree);
        }
        updateBaseFValues(b);
      }else{
        F = best_F;
        additive_trees[m].swap(best_trees);
      }
    }else{
      int b = base_classes[m];
      computeHessianResidual(b);
      for (int k = 0; k < K; ++k) {
        if (k == b) continue;
        zeroBins();
        Tree *tree;
        tree = new Tree(data, config);
        tree->init(&hist, &buffer[0], &buffer[1], &feature_importance,
                   &(hessians[k * data->n_data]), &(residuals[k * data->n_data]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
        tree->buildTree(&ids, &fids);
        updateF(k, tree);
        additive_trees[m][k] = std::unique_ptr<Tree>(tree);
      }
      updateBaseFValues(b);
    }

    double loss = getLoss();
    if ((m + 1) % config->model_eval_every == 0){
      print_train_message(m + 1,loss,t1.get_time_restart());
    }
    if (m < config->model_n_iterations - 1)
      base_classes[m + 1] = base_classes[m];
    if (config->save_model && (m + 1) % config->model_save_every == 0) {
      saveModel(m + 1);
    }
    if(loss < config->stop_tolerance){
      config->model_n_iterations = m + 1;
      break;
    }
  }
  printf("Training has taken %.5f seconds\n", t2.get_time());

  if (config->save_model) saveModel(config->model_n_iterations);
  getTopFeatures();
}

/**
 * Method to implement testing process for ABCMART algorithm as described by
 * Li Ping et Al. (2008). Descriptions for used sub-routines are available
 * in the individual method-comments.
 */
void ABCMart::test() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  Utils::Timer t1;
  t1.restart();

  double best_accuracy = 0;
  int K = data->data_header.n_classes;
  int low_err = std::numeric_limits<int>::max();
  for (int m = 0; m < config->model_n_iterations; ++m) {
    for (int k = 0; k < K; ++k) {
      if (additive_trees[m][k] != NULL) {
        additive_trees[m][k]->init(nullptr, &buffer[0], &buffer[1], nullptr,
                                   nullptr, nullptr,ids_tmp.data(),H_tmp.data(),R_tmp.data());
        std::vector<double> updates = additive_trees[m][k]->predictAll(data);
        
        if (m < config->warmup_iter){
          for (int i = 0; i < data->n_data; ++i) {
            F[k][i] += config->model_shrinkage * updates[i];
          }
        }else{
          int b = base_classes[m];
          for (int i = 0; i < data->n_data; ++i) {
            F[k][i] += config->model_shrinkage * updates[i];
            F[b][i] -= config->model_shrinkage * updates[i];
          }
        }
      }
    }

    if ((m + 1) % config->model_eval_every == 0){
      print_test_message(m + 1,t1.get_time_restart(),low_err);
    }
  }
}

/**
 * Method to implement training process for ABCMART algorithm as described
 * by Ping Li (2008). Descriptions for used sub-routines are available
 * in the individual method-comments.
 * @param[in] start: index to start training at.
 */
void ABCMart::train_worst() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  int K = data->data_header.n_classes;

  Utils::Timer t1, t2, t3;
  t1.restart();
  t2.restart();
  double res = 0;

  if(start_epoch != 0){
    base_classes[start_epoch] = argmax(class_losses);
  }

  for (int m = start_epoch; m < config->model_n_iterations; m++) {
    if (config->model_data_sample_rate < 1) {
      ids = sample(data->n_data, config->model_data_sample_rate);
    }
    if (config->model_feature_sample_rate < 1) {
      fids =
          sample(data->data_header.n_feats, config->model_feature_sample_rate);
    }
    int b = base_classes[m];
    computeHessianResidual(b);
    for (int k = 0; k < K; ++k) {
      if (k == b) {
        additive_trees[m][k] = std::unique_ptr<Tree>(nullptr);
        continue;
      }
      zeroBins();
      Tree *tree;
      tree = new Tree(data, config);
      tree->init(&hist, &buffer[0], &buffer[1], &feature_importance,
                 &(hessians[k * data->n_data]), &(residuals[k * data->n_data]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
      tree->buildTree(&ids, &fids);
      tree->updateFeatureImportance(m);
      updateF(k, tree);
      additive_trees[m][k] = std::unique_ptr<Tree>(tree);
    }
    updateBaseFValues(b);

    double loss = getLoss();
    if ((m + 1) % config->model_eval_every == 0)
      printf("%4d | loss: %20.14e | acc: %.4f | time: %.5f\n", m + 1,
             loss, getAccuracy(), t1.get_time_restart());
    if (m < config->model_n_iterations - 1) {
      base_classes[m + 1] = argmax(class_losses);
    }
    if (config->save_model && (m + 1) % config->model_save_every == 0) saveModel(m + 1);
  }
  printf("Training has taken %.5f seconds\n", t2.get_time());

  if (config->save_model) saveModel(config->model_n_iterations);
  getTopFeatures();

}

/**
 * Helper method to compute hessian and residual simultaneously for ABCMart.
 * @param b : the current base class
 */
void ABCMart::computeHessianResidual(int b) {
  std::vector<double> prob;
#pragma omp parallel for schedule(static) private(prob)
  for (unsigned int i = 0; i < data->n_data; ++i) {
    prob.resize(data->data_header.n_classes);
    int label = int(data->Y[i]);
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      prob[k] = F[k][i];
    }
    softmax(prob);
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      double p_ik = prob[k];
      double p_ib = prob[b];
      int r_ik = (int)(k == label);  // indicators used in calculations
      int r_ib = (int)(b == label);
      residuals[k * data->n_data + i] = p_ib - r_ib + r_ik - p_ik;
      hessians[k * data->n_data + i] = p_ik * (1 - p_ik) + p_ib * (1 - p_ib) + 2 * p_ib * p_ik;
    }
  }
}

void ABCMart::computeHessianResidual() {
  std::vector<double> prob;
#pragma omp parallel for schedule(static) private(prob)
  for (unsigned int i = 0; i < data->n_data; ++i) {
    prob.resize(data->data_header.n_classes);
    int label = int(data->Y[i]);
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      prob[k] = F[k][i];
    }
    softmax(prob);
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      double p_ik = prob[k];
      residuals[k * data->n_data + i] = (k == label) ? (1 - p_ik) : -p_ik;
      hessians[k * data->n_data + i] = p_ik * (1 - p_ik);
    }
  }
}

void ABCMart::computeHessianResidual(int b,std::vector<unsigned int>& abc_sample_ids){
  std::vector<double> prob;
#pragma omp parallel for schedule(static) private(prob)
  for (unsigned int ii = 0;ii < abc_sample_ids.size();++ii) {
    int i = abc_sample_ids[ii];
    prob.resize(data->data_header.n_classes);
    int label = int(data->Y[i]);
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      prob[k] = F[k][i];
    }
    softmax(prob);
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      double p_ik = prob[k];
      double p_ib = prob[b];
      int r_ik = (int)(k == label);  // indicators used in calculations
      int r_ib = (int)(b == label);
      residuals[k * data->n_data + i] = p_ib - r_ib + r_ik - p_ik;
      hessians[k * data->n_data + i] = p_ik * (1 - p_ik) + p_ib * (1 - p_ib) + 2 * p_ib * p_ik;
    }
  }
}
/**
 * Helper method to compute CE loss on current probabilities and populate
 * class losses for proceeding base_class suggestion.
 * @return summed CE loss over training set.
 */
double ABCMart::getLoss() {
  double loss = 0.0;
  //for (int i = 0; i < class_losses.size(); ++i) class_losses[i] = 0;
 
  std::vector<double> local_class_losses(class_losses.size());
#ifndef OS_WIN
  #pragma omp parallel for reduction(+: loss) reduction(vec_double_plus: local_class_losses)
#endif
  for (int i = 0; i < data->n_data; i++) {
    int y = int(data->Y[i]);
    if (y < 0) continue;
    double curr = F[y][i];
    double denominator = 0;

    for (int k = 0; k < data->data_header.n_classes; ++k) {
      double tmp = F[k][i] - curr;
      if (tmp > 700) tmp = 700;
      denominator += exp(tmp);
    }
    double loss_i = -log(1.0 / denominator);
    loss += loss_i;
    local_class_losses[y] += loss_i;
  }
  class_losses = local_class_losses;
  return loss;
}

/**
 * Helper method to load the pre-trained model.
 * @return final training iteration of loaded model.
 */
int ABCMart::loadModel() {
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

void ABCMart::init() {
  GradientBoosting::init();
  base_classes.resize(config->model_n_iterations, 0);
  class_losses.resize(data->data_header.n_classes, 0.0);
  if(config->model_mode == "train" && start_epoch == 0){
    std::vector<int> bins(data->data_header.n_classes, 0);
    for (double y : data->Y) {
      bins[int(y)]++;
      class_losses[int(y)] += 1;
    }

    int maxCount = 0;
    for (int i = 0; i < bins.size(); ++i) {
      if (bins[i] > maxCount) {
        maxCount = bins[i];
        base_classes[0] = i;
      }
    }
  }
}

/**
 * Helper method to save the current model.
 */
void ABCMart::saveModel(int iter) {
  FILE *model_out =
      fopen((experiment_path + config->model_suffix).c_str(), "wb");
  if (model_out == NULL) {
    fprintf(stderr, "[ERROR] Cannot create file: (%s)\n",
            (experiment_path + config->model_suffix).c_str());
    exit(1);
  }
  ModelHeader model_header;
  model_header.config = *config;
  model_header.config.model_n_iterations = iter;

  model_header.auxDataHeader = data->data_header;
  model_header.serialize(model_out);
  serializeTrees(model_out, iter);
  // save trees
  for (int i = 0; i < iter; ++i) {
    fwrite(&(base_classes[i]), sizeof(int), 1, model_out);
  }
  Utils::serialize_vector(model_out, class_losses);
  fclose(model_out);
}

/**
 * Update the f values for the base class.
 * @param[in] b : the base class index.
 */
void ABCMart::updateBaseFValues(int b) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < data->n_data; ++i) {
    double base_f = 0.0;
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      if (k == b) continue;  // skip base class, as value is being calculated
      base_f -= F[k][i];
    }
    F[b][i] = base_f;
  }
}

double ABCMart::getLossRaw(int b,const std::vector<unsigned int>& ids) const{
  double loss = 0.0;
  #pragma omp parallel for schedule(static) reduction(+: loss) 
  for (int ii = 0; ii < ids.size(); ++ii) {
    int i = ids[ii];
    double base_f = 0.0;
    int y = int(data->Y[i]);
    if (y < 0) continue;
    double curr = F[y][i];
    double denominator = 0;
    for (int k = 0; k < data->data_header.n_classes; ++k) {
      if(k == b)
        continue;
      base_f -= F[k][i];
      double tmp = F[k][i] - curr;
      if (tmp > 700) tmp = 700;
      denominator += exp(tmp);
    }
    double tmp = base_f - curr;
    if (tmp > 700) tmp = 700;
    denominator += exp(tmp);
    double loss_i = -log(1.0 / denominator);
    loss += loss_i;
  }
  return loss;
}


LambdaMart::LambdaMart(Data *data, Config *config) : GradientBoosting(data, config) {
  data->loadRankQuery();
}

void LambdaMart::savePrediction(){
  GradientBoosting::savePrediction();
}

void GradientBoosting::print_rank_test_message(int iter,double iter_time){
  if(config->no_label)
    return;
  auto p = getNDCG();
  double NDCG = p.second;
  printf("%4d | NDCG: %20.14e | time: %.5f\n", iter,
       NDCG, iter_time);
#ifdef USE_R_CMD
 R_FlushConsole();
#endif
  if(config->save_log){
    fprintf(log_out,"%4d %20.14e %.5f\n", iter, NDCG, iter_time);
  }
}

/**
 * Method to implement testing process for LambdaMART algorithm as described by
 * Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 */
void LambdaMart::test() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();
  
  Utils::Timer t1;
  t1.restart();

  int K = 1;
  for (int m = 0; m < config->model_n_iterations; ++m) {
    for (int k = 0; k < K; ++k) {
      if (additive_trees[m][k] != NULL) {
        additive_trees[m][k]->init(nullptr, &buffer[0], &buffer[1],
                                   nullptr, nullptr, nullptr,ids_tmp.data(),H_tmp.data(),R_tmp.data());
        std::vector<double> updates =
            additive_trees[m][k]->predictAll(data);
        for (int i = 0; i < data->n_data; ++i) {
          F[k][i] += config->model_shrinkage * updates[i];
        }
      }
    }
    print_rank_test_message(m + 1,t1.get_time_restart());
  }
}


/**
 * Method to implement training process for MART algorithm as described
 * by Friedman et Al. (2001). Descriptions for used sub-routines are available
 * in the individual method-comments.
 * @param[in] start: index to start training at.
 */
void LambdaMart::train() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  // build one tree if it is binary prediction
  int K = 1;

  Utils::Timer t1, t2;
  t1.restart(); t2.restart();
  for (int m = start_epoch; m < config->model_n_iterations; m++) {
    computeHessianResidual();
    for (int k = 0; k < 1; ++k) {
      zeroBins();
      Tree *tree;
      tree = new Tree(data, config);
      tree->init(&hist, &buffer[0], &buffer[1],
            &feature_importance, &(hessians[k * data->n_data]), &(residuals[k * data->n_data]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
      tree->buildTree(&ids, &fids);
      updateF(k, tree);
      additive_trees[m][k] = std::unique_ptr<Tree>(tree);
    }
    auto p = getNDCG();
    print_rank_train_message(m + 1,p.second,t1.get_time_restart());
    if (config->save_model && (m+1) % config->model_save_every == 0) saveModel(m+1);
  }
  printf("Training has taken %.5f seconds\n", t2.get_time());

  if (config->save_model) saveModel(config->model_n_iterations);
  getTopFeatures();

}

void GradientBoosting::print_rank_train_message(int iter,double NDCG,double iter_time){
  printf("%4d | NDCG: %20.14e | time: %.5f\n", iter,
       NDCG, iter_time);
#ifdef USE_R_CMD
  R_FlushConsole();
#endif
  if(config->save_log)
    fprintf(log_out,"%4d %20.14e %.5f\n", iter, NDCG, iter_time);
}

/**
 * Helper method to compute hessian and residual simultaneously.
 */
void LambdaMart::computeHessianResidual() {
  const int TOPK_NDCG = 1;
  double sigma = 2.0;
  double loss = 0;
  double total_NDCG = 0;
  double total_topk_NDCG = 0;
  
  #pragma omp parallel for reduction(+: total_NDCG) schedule(guided)
  for(size_t omp_iter = 0;omp_iter < data->rank_groups.size();++omp_iter){
    const auto& p = data->rank_groups[omp_iter];
    const auto& start = p.first;
    const auto& end = p.second;
    std::vector<double> pred;
    pred.reserve(end - start + 1);
    for(auto i = start;i < end;++i)
        pred.push_back(F[0][i]);
    std::vector<int> idx(pred.size());
    for(int i = 0;i < idx.size();++i)
        idx[i] = i;
    auto idx_max = idx;
    std::sort(idx_max.begin(),idx_max.end(),[start,this](int a,int b){return data->Y[start + a] > data->Y[start + b];});
    double max_DCG = 0;
    double topk_max_DCG = 0;
    for(int i = 0;i < idx_max.size();++i){
      max_DCG += ((1 << (int)data->Y[start + idx_max[i]]) - 1) / log2(2 + i);
      if(i < TOPK_NDCG)
        topk_max_DCG += ((1 << (int)data->Y[start + idx_max[i]]) - 1) / log2(2 + i);
    }
    std::sort(idx.begin(),idx.end(),[&pred](int a, int b) {return pred[a] > pred[b];});
    std::vector<double> base_DCG(idx.size());
    std::vector<int> idx2rank(idx.size());
    double DCG = 0;
    double topk_DCG = 0;
    for(int i = 0;i < idx.size();++i){
      base_DCG[idx[i]] = ((1 << (int)data->Y[start + idx[i]]) - 1) / log2(2 + i);
      idx2rank[idx[i]] = i;
      DCG += base_DCG[idx[i]]; 
      if(i < TOPK_NDCG)
        topk_DCG += base_DCG[idx[i]]; 
      if(max_DCG >= 1e-10)
        total_NDCG += base_DCG[idx[i]] / max_DCG;
    }
    if(topk_max_DCG < 1e-10){
      total_topk_NDCG += 1;
    }else{
      total_topk_NDCG += topk_DCG / topk_max_DCG;
    }
    if(max_DCG < 1e-10){
      total_NDCG += 1;
      for(auto i = start;i < end;++i){
        residuals[i] = 0;
        hessians[i] = 0;
      }
      continue;
    }
    for(auto i = start;i < end;++i){
      residuals[i] = 0;
      hessians[i] = 0;
    }
    for(auto i = start;i < end;++i){
      double lambdai = 0;
      double hessiani = 0;
      for(auto j = start;j < end;++j){
        if(i == j || (int)(data->Y[i]) <= (int)(data->Y[j]))
          continue;
        double delta_DCG = ((1 << (int)data->Y[i]) - (1 << (int)data->Y[j])) * (1 / log2(2 + idx2rank[i - start]) - 1/log2(2 + idx2rank[j - start]));      
        double delta_NDCG = delta_DCG / max_DCG;
        double oij = F[0][i] - F[0][j];
        double sij = (data->Y[i] - data->Y[j]) > 0 ? 1 : -1; 
        double pij = 1 / (1 + exp(sigma * oij));
        double lambdaij = sigma * sij * fabs(delta_NDCG * pij); 

        double delta_hessian = pij * (1 - pij);
        double delta_loss = 0; 
        lambdai += lambdaij;
        hessiani += sigma * sigma * fabs(delta_NDCG) * delta_hessian;
        residuals[j] -= lambdaij;
        hessians[j] += delta_hessian;
      }
      residuals[i] += lambdai;
      hessians[i] += hessiani;
    }
  }
}


std::pair<double,double> GradientBoosting::getNDCG(){
  if(data->rank_groups.size() == 0 || data->rank_groups[data->rank_groups.size() - 1].second != data->n_data){
    fprintf(stderr,"[Error] query file does not match data!\n");
    exit(1);
  }

  double total_NDCG = 0;
  int zero_groups = 0;
  for(const auto& p : data->rank_groups){
    const int start = p.first;
    const int end = p.second;
    std::vector<double> curr_score;
    for(int i = start;i < end;++i){
      curr_score.push_back(F[0][i]);
    }
    auto NDCG = Utils::RankUtils::computeNDCG(curr_score,data->Y,start);
    if(NDCG < 1e-10)
      ++zero_groups;
    else
      total_NDCG += NDCG;
  }

  auto avgNDCG_count0 = total_NDCG / (data->rank_groups.size() - zero_groups);
  auto avgNDCG_count1 = (total_NDCG + zero_groups) / data->rank_groups.size();
  return std::make_pair(avgNDCG_count0,avgNDCG_count1);
}

GBRank::GBRank(Data* data, Config* config) : GradientBoosting(data,config){
  tau =config->gbrank_tau;
  tau2 = config->gbrank_update_factor * tau;
}

void GBRank::train(){
  data->loadRankQuery();
  // set up buffers for OpenMP
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();
  // build one tree if it is binary prediction
  int K = 1;

  Utils::Timer t1, t2;
  t1.restart(); t2.restart();
  config->model_use_logit = true;
  for (int m = start_epoch; m < config->model_n_iterations; m++) {
    computeHessianResidual();
    zeroBins();
    Tree *tree = new Tree(data, config);
    tree->init(&hist, &buffer[0], &buffer[1],
               &feature_importance, &(hessians[0]), &(residuals[0]),ids_tmp.data(),H_tmp.data(),R_tmp.data());
    tree->buildTree(&ids, &fids);
    GBupdateF(0, tree,m + 1);
		additive_trees[m][0] = std::unique_ptr<Tree>(tree);
    auto p = getNDCG();
    print_rank_train_message(m + 1,p.second,t1.get_time_restart());
    if (config->save_model && (m+1) % config->model_save_every == 0) saveModel(m+1);
  }
  printf("Training has taken %.5f seconds\n", t2.get_time());

  if (config->save_model) saveModel(config->model_n_iterations);
  getTopFeatures();
}

void GBRank::computeHessianResidual() {
  std::vector<double> gb_label(data->n_data,0.0);
  std::vector<int> gb_label_cnt(data->n_data,0);
  int negs_cnt = 0;
	#pragma omp parallel for schedule(guided)
  for(int pp = 0;pp < data->rank_groups.size();++pp){
		const auto& p = data->rank_groups[pp];
    const auto& start = p.first;
    const auto& end = p.second;
    for(int i = start;i < end;++i){
      for(int j = start;j < end;++j){
        if(F[0][i] < F[0][j] + tau && data->Y[i] > data->Y[j]){
          gb_label[i] += F[0][j] + tau2;
          ++gb_label_cnt[i];
          gb_label[j] += F[0][i] - tau2;
          ++gb_label_cnt[j];
          ++negs_cnt;
        }
      }
    }
  }
  #pragma omp parallel for schedule(static) if (config->use_omp)
  for(int i = 0; i < data->n_data; i++) {
    residuals[i] = gb_label_cnt[i] != 0 ? -1.0*(-gb_label[i]) : 0;
    hessians[i] = gb_label_cnt[i] != 0 ? 1 : 0;
  }
}


void GBRank::GBupdateF(int k, Tree *tree,int n_iter) {
  std::vector<unsigned int> &ids = tree->ids;
  std::vector<double> &f = F[k];
  for (int lf = 0; lf < tree->n_leaves; ++lf) {
    int leaf_id = tree->leaf_ids[lf];
    if(leaf_id < 0)
        continue;
    Tree::TreeNode node = tree->nodes[leaf_id];
    double update = config->model_shrinkage * node.predict_v;
    unsigned int start = node.start, end = node.end;
    #pragma omp parallel for if (config->use_omp)
    for (int i = start; i < end; ++i)
      f[ids[i]] = (f[ids[i]] * n_iter + update) / (n_iter + 1);
  }
  tree->freeMemory();
}


void GBRank::test() {
  std::vector<std::vector<std::vector<unsigned int>>> buffer =
      GradientBoosting::initBuffer();

  Utils::Timer t1;
  t1.restart();

  data->loadRankQuery();
  for (int m = 0; m < config->model_n_iterations; m++) {
    if (additive_trees[m][0] != NULL) {
      additive_trees[m][0]->init(nullptr, &buffer[0], &buffer[1],
                                   nullptr, nullptr, nullptr,ids_tmp.data(),H_tmp.data(),R_tmp.data());
      std::vector<double> updates = additive_trees[m][0]->predictAll(data);
      for (int i = 0; i < data->n_data; i++) {
        F[0][i] = ((m + 1) * F[0][i] + updates[i] * config->model_shrinkage) / (m + 2);
      }
    }
    print_rank_test_message(m + 1,t1.get_time_restart());
  }
}


void GBRank::savePrediction(){
	GradientBoosting::savePrediction();
}

}  // namespace ABCBoost
