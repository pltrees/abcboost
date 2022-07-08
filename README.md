# ABCBoost

This toolkit consists of ABCBoost, a concrete implementation of [Fast ABCBoost](https://arxiv.org/pdf/2205.10927.pdf) (Fast Adaptive Base Class Boost). 

## Quick Start
### Installation guide
Run the following commands to build ABCBoost from source:
```
git clone git@github.com:pltrees/abcboost.git
cd abcboost
mkdir build
cd build
cmake ..
make
cd ..
```
This will create three executables (`train`, `predict` and `map`) in the `abcboost` directory.
`train` is the executable to train models.
`predict` is the executable to validate and inference using trained models.
`map` is an auxiliary executable to generate the histogram mapping without actually training the model. The mapping is used to quantize the real-value features to negative integers within 0..`-data_max_n_bins`.

The default setting builds ABCBoost with multi-thread support [OpenMP](https://en.wikipedia.org/wiki/OpenMP) (OpenMP comes with the system GCC toolchains on Linux).
To turn off the multi-thread option, set `OMP=OFF`:
```
cmake -DOMP=OFF ..
make clean
make
```


To build ABCBoost with GPU support, install [NVIDIA CUDA Toolkits](https://developer.nvidia.com/cuda-downloads) and set the option `CUDA=ON`:
```
cmake -DOMP=ON -DCUDA=ON ..
make
```

### Training

Place your data (either in LIBSVM format or CSV Matrix format, here CSV-format delimiter can be comma, space, or tab) under the `abcboost` folder. For example, we have put 5 example datasets: [M-Image](http://www.dumitru.ca/files/publications/icml_07.pdf) for classification, [zip](https://hastie.su.domains/ElemStatLearn/data.html) for classiciation on sparse data, [ijcnn1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1) in for binary classification, [comp_cpu](http://www.cs.toronto.edu/~delve/data/comp-activ/desc.html) for regression, and [mslr10k](https://www.microsoft.com/en-us/research/project/mslr/) a small subset for illustrating the ranking task into the `data` folder.

We start with M-Image as our example. It is compressed by bzip2.
Unzip it by:
```
bunzip2 data/mimage*.bz
```
Note that only the train file is required for training. To train the model with 100 iterations, launch `train` from the command line:
```
./train -data data/mimage.train.libsvm -J 20 -v 0.1 -iter 100
```
where `-J 20` specifies 20 terminal node for each tree, `-v 0.1` represents the shrinkage rate, a.k.a., learning rate, is 0.1.
Two files are generated in the working directory:
* The saved model we just trained: `mimage.train.libsvm_abcrobustlogit_J20_v0.1.model`
* The mapping file of the mimage.train.libsvm data: `mimage.train.libsvm.map`

If the executables are compiled with GPU support, we can specify the GPU device from the command line:
```
CUDA_VISIBLE_DEVICES=0 ./train -data data/mimage.train.libsvm -J 20 -v 0.1 -iter 100
```
In the above example, we specify `GPU 0` as the device. (Use `nvidia-smi` to find out available GPUs)

#### Other Training Methods
<strong>Classification:</strong>
The default training methods of ABCBoost is `abcrobustlogit` for classification problems. Other methods as `mart`, `robustlogit` and `abcmart` are supported. Use `-method` option to specify other methods. For example:
```
./train -data data/mimage.train.libsvm -J 20 -v 0.1 -iter 100 -method mart
```
<strong>Regression:</strong>
Regression problems are supported by adding `-method regression`:
```
./train -data data/comp_cpu.train.libsvm -J 20 -v 0.1 -iter 100 -method regression
```

<strong>Ranking:</strong>
Ranking tasks are supported by using `-method lambdarank`. Note that the query/group file need to be specified (the query file tells us how many instances in the data for each query):
```
./train -data data/mslr10k.train -query data/mslr10k.train.query -J 20 -v 0.1 -iter 100 -method lambdarank
./predict -data data/mslr10k.test -query data/mslr10k.test.query -model mslr10k.train_lambdarank_J20_v0.1.model
```

### Testing

To test the model with the test data `mimage.test.libsvm`, type the below in command line:
```
./predict -data data/mimage.test.libsvm -model mimage.train.libsvm_abcrobustlogit_J20_v0.1.model
```

To test the first 50 iterations of the trained model (the trained model has 100 iterations):
```
./predict -data mimage.test.libsvm -model mimage.train.libsvm_abcrobustlogit_J20_v0.1.model -iter 50
```
### Parameter Tuning

Here we illustrate some common parameters and provide some examples:
* `-iter` number of iterations (default 1000)
* `-J` number of leaves in a tree (default 20)
* `-v` learning rate (default 0.1)
* `-search` searching size for the base class (default 1: we greedily choose the base classes according to the training loss). For example, 2 means we try the class with the greatest loss and the class with the second greatest loss as base class and pick the one with lower loss as the base class for the current iteration.
* `-n_threads` number of threads (default 1) <strong>It can only be used when multi-thread is enabled. (Compile the code with `-DOMP=ON` in cmake.)</strong>
* `-additional_files` using other files to do bin quantization besides the training data. File names are separated by `,` without additional spaces, e.g., `-additional_files file1,file2,file3`.

To train the model with 2000 iterations, 16 leaves per tree and 0.08 learning rate:
```
./train -data data/mimage.train.libsvm -iter 2000 -J 16 -v 0.08
```

To train the model with 2000 iterations, 16 leaves per tree, 0.08 learning rate and enable the exhaustive base class searching (10 is the number of classes of the mimage.train.libsvm dataset):
```
./train -data data/mimage.train.libsvm -iter 2000 -J 16 -v 0.08 -search 10
```
Note that the exhaustive searching produces better-generalized model while requiring substantially more time.

To train the model with 2000 iterations, 16 leaves per tree, 0.08 learning rate and also use the value of the test data to quantize the data:
```
./train -data data/mimage.train.libsvm -iter 2000 -J 16 -v 0.08 -additional_files data/mimage.test.libsvm
```
The labels in the specified additional files are not used in the training. Only the feature values are used to generate (potentially) better quantization. Better testing results may be obtained when using additional files



## More Configuration Options:
#### Data related:
* `-data_use_mean_as_missing`
* `-data_min_bin_size` minimum size of the bin
* `-data_sparsity_threshold`
* `-data_max_n_bins` max number of bins (default 1000)
* `-data_path, -data` path to train/test data
#### Tree related:
* `-tree_clip_value` gradient clip (default 50)
* `-tree_damping_factor`, regularization on numerator (default 1e-100)
* `-tree_max_n_leaves`, -J (default 20)
* `-tree_min_node_size` (default 10)
#### Model related:
* `-model_use_logit`, whether use logitboost
* `-model_data_sample_rate` (default 1.0)
* `-model_feature_sample_rate` (default 1.0)
* `-model_shrinkage`, `-shrinkage`, `-v`, the learning rate (default 0.1)
* `-model_n_iterations`, `-iter` (default 1000)
* `-model_save_every`, `-save` (default 100)
* `-model_eval_every`, `-eval` (default 1)
* `-model_name`, `-method` regression/lambdarank/mart/abcmart/robustlogit/abcrobustlogit (default abcrobustlogit)
* `-model_pretrained_path`, `-model`
#### Adaptive Base Class (ABC) related:
* `-model_base_candidate_size`, `base_candidates_size`, `-search` (default 2) base class searching size in abcmart/abcrobustlogit
* `-model_gap`, `-gap` (default 5) the gap between two base class searchings. For example, `-model_gap 2` means we will do the base class searching in iteration 1, 4, 6, ...
* `-model_warmup_iter`, `-warmup_iter` (default 0) the number of iterations that use normal boosting before ABC method kicks in. It might be helpful for datasets with a large number of classes when we only have a limited base class searching parameter (`-model_base_candidate_size`) 
* `-model_warmup_use_logit`, `-warmup_use_logit` 0/1 (default 1) whether use logitboost in warmup iterations.
* `-model_abc_sample_rate`, `-abc_sample_rate` (default 1.0) the sample rate used for the base class searching
* `-model_abc_sample_min_data` `-abc_sample_min_data` (default 2000) the minimum sampled data for base class selection. This parameter only takes into effect when `-abc_sample_rate` is less than `1.0`
#### Regression related:
* `-regression_lp_loss`, `-lp` (default 2.0) whether use Lp norm instead of L2 norm. p (p >= 1.0) has to be specified
* `-regression_use_hessian` 0/1 (default 1) whether use second-order derivatives in the regression. This parameter only takes into effect when `-regression_lp_loss p` is set and `p` is greater than `2`.
* `-regression_huber_loss`, `-huber` 0/1 (default 0) whether use huber loss
* `-regression_huber_delta`, `-huber_delta` the delta parameter for huber loss. This parameter only takes into effect when `-regression_huber_loss 1` is set
#### Parallelism:
* `-n_threads`, `-threads` (default 1)
* `-use_gpu` 0/1 (default 1 if compiled with CUDA) whether use GPU to train models. This parameter only takes into effect when the executable is complied with CUDA (i.e., the flag `-DCUDA=on` is enabled in `cmake`).
#### Other:
* `-save_log`, 0/1 (default 0) whether save the runtime log to file
* `-save_model`, 0/1 (default 1)
* `-no_label`, 0/1 (default 0) It should only be enabled to output prediction file when the testing data has no label and `-model_mode` is `test`
* `-stop_tolerance` (default 2e-14) It works for all non-regression tasks, e.g., classification. The training will stop when the total training loss is less than the stop tolerance.
* `-regression_stop_factor` (default 1e-5) The auto stopping criterion is different from the classification task because the scale of the regression target is unknown. We adaptively set the regression stop tolerate to `regression_stop_factor * total_loss / sum(y^p)`, where `y` is the regression targets and `p` is the value specified in `-regression_lp_loss`.
* `-regression_auto_clip_value` 0/1 (default 1) whether use our adaptive clipping value computation for the predict value on terminal nodes. When enabled, the adaptive clipping value is computed as `tree_clip_value * max_y - min_y` where `tree_clip_value` is set via `-tree_clip_value`, `max_y` and `min_y` are the maximum and minimum regression target value, respectively.
## R Support
We provide an R library to enable calling ABCBoost subroutines from R.
To build and install the library, type the following command in `abcboost/`:
```
cd ..
R CMD build abcboost
R CMD INSTALL abcboost_1.0.0.tar.gz
```
We also provide a pre-built `abcboost_1.0.0.tar.gz` in `abcboost/` for convenience.
R CMD INSTALL may require sudo privilege if your R is installed in the system global path.
To install it in a local path, say `abcboost/lib`
```
mkdir abcboost/lib
R CMD INSTALL -l abcboost/lib abcboost_1.0.0.tar.gz
```
For the local installment, we need to use `library(abcboost,lib.loc=abcboost/lib)` instead of simply `library(abcboost)`.


Function signature:
```
abcboost_train <- function(train_Y,train_X,model_name,iter,leaves,shinkage,search = 1,gap = 0,params=NULL){
abcboost_test <- function(test_Y,test_X,model){
abcboost_save_model <- function(model,path){
abcboost_load_model <- function(path){
```
Here we show an example of training and testing:
```
library(abcboost)
data <- read.csv(file='data/zip.train.csv',header=FALSE)
X <- data[,-1]
Y <- data[,1]
data <- read.csv(file='data/zip.test.csv',header=FALSE)
testX <- data[,-1]
testY <- data[,1]
# The last argument of abcboost_train is optional. 
# We use n_threads as an example
# All command line supported parameters can be passed via list: 
# list(parameter1=value1, parameter2=value2,...)
model <- abcboost_train(Y,X,"abcrobustlogit",100,20,0.1,3,0,list(n_threads=1))
# abcboost_save_model(model,'mymodel.model')
# model <- abcboost_load_model('mymodel.model')
res <- abcboost_test(testY,testX,model)
# We also provide a method to read libsvm format data into sparse array
data <- abcboost_read_libsvm('data/zip.train.libsvm')
X <- data$X
Y <- data$Y
data <- abcboost_read_libsvm('data/zip.train.libsvm')
testX <- data$X
testY <- data$Y
# X can be a either a dense matrix or a sparse matrix
# The interface is the same as the dense case, 
# but with better performance for sparse data
model <- abcboost_train(Y,X,"abcrobustlogit",100,20,0.1,3,0,list(n_threads=1))
res <- abcboost_test(testY,testX,model)
```

## Matlab Support
We provide a Matlab wrapper to call ABCBoost subroutines from Matlab.
To compile the Matlab mex files in Matlab:
```
cd src/matlab
compile
```
Please make sure Matlab is installed and `mex` can be executed by typing `mex` from the Matlab command line.

After the compilation, here we show an example of training and testing:
```
tr = load('../../data/zip.train.csv');
te = load('../../data/zip.test.csv');
Y = tr(:,1);
X = tr(:,2:end);
testY = te(:,1);
testX = te(:,2:end);

params = struct;
params.n_threads = 1;
% The params argument is optional. 
% We use n_threads as an example
% All command line supported parameters can be passed via params: params.parameter_name = value
model = abcboost_train(Y,X,'abcrobustlogit',100,20,0.1,1,0,params);
% abcboost_save(model,'mymodel.model');
% model = abcboost_load('mymodel.model');
res = abcboost_test(testY,testX,model);
% Sparse matlab matrix is also supported
% For example, we included the libsvmread.c from the LIBSVM package for data loading
% We need to compile it before actual usage:
% mex libsvmread.c  
[Y, X] = libsvmread('../../data/zip.train.libsvm');
[testY, testX] = libsvmread('../../data/zip.train.libsvm');
% Here X and testX are sparse matrices
model = abcboost_train(Y,X,'abcrobustlogit',100,20,0.1,1,0,params);
res = abcboost_test(testY,testX,model);
```

## Python Support
We provide the python support through `pybind11`.
Before the compilation, `pybind11` should be installed:

`python3 -m pip install pybind11`

To compile:
```
cd abcboost
bash compile_py.sh
```
After the compilation, a shared library `abcboost.so` is generated.

Make sure `abcboost.so` is in the current directory.
Paste the following code in a `python3` interactive shell:
```
import numpy as np
# We use a matrix-format sample data here
data = np.genfromtxt('data/zip.train.csv',delimiter=',').astype(float)
#
Y = data[:,0]
X = data[:,1:]
data = np.genfromtxt('data/zip.test.csv',delimiter=',').astype(float)
testY = data[:,0]
testX = data[:,1:]
import abcboost
model = abcboost.train(Y,X,'abcrobustlogit',100,20,0.1,3,0)
# All command line supported parameters can be passed as optional keyword arguments
# For example:
# model = abcboost.train(Y,X,'abcrobustlogit',100,20,0.1,3,0,n_threads=1)
res = abcboost.test(Y,X,model)
# abcboost.save(model,'mymodel.model')
# model = abcboost.load('mymodel.model')
res = abcboost.test(testY,testX,model)
# Alternatively, we also support libsvm-format sparse matrix
# We use sklearn to load libsvm format data as a scipy.sparse matrix
# sklearn can be installed as: python3 -m pip install scikit-learn
import sklearn
import sklearn.datasets
# X is a scipy.sparse matrix
[X, Y] = sklearn.datasets.load_svmlight_file('data/zip.train.libsvm')
[testX, testY] = sklearn.datasets.load_svmlight_file('data/zip.train.libsvm')
# The training and testing interfaces are unified for both dense and sparse matrices
model = abcboost.train(Y,X,'abcrobustlogit',100,20,0.1,3,0)
res = abcboost.test(testY,testX,model)
```

## Feature Highlights
### Application
We support the following learning tasks:

* Regression
* Classification
* Ranking

### Adaptive base class
Two key components of ABCBoost include:

* With the sum-to-zero constraint on the loss function, we only build trees for `K − 1` classes and save computation;

* We adaptively choose the current `s` "worst" class as the base class at each boosting step, where `s` is 1 in default and can be specified with `-search` option, e.g., `-search 10` searches `10` "worst" classes for the best base class.

When computing the base base, we either support heuristic search strategy or introduce gaps (where we update the choice of the base class only for every g+1 boosting steps) to speed up learning.

### Feature ranking
We provide individual importance calculation for each input feature to help understand the model. Feature importance is calculated as the sum of information gains at all splits associated with the feature of interest.

### Stochastic training
We provide the option which selects a random sample of the training set at each boosting step and fits trees to these samples. We also support feature bagging which selects a random subset of the features at each candidate split in the learning process. Sampling rates can be set with the `-model_data_sample_rate` and `-model_feature_sample_rate`parameter.

### Restoring saved models
One can restore a pre-trained model and use it for prediction, fine-tuning or further training with the `-model_pretrained_path` parameter or `-model` for short. For instance,
```
./predict -data data/mimage.train.libsvm -model mimage.train.libsvm_abcrobustlogit_J16_v0.08.model
```

## Parallelism for Large Datasets
We provide parallelism for the program on both multi-core machines ([OpenMP](https://www.openmp.org)) and machines with GPU
([CUDA](https://developer.nvidia.com/about-cuda)). The program will use parallelism as default and automatically detect the number of threads on the machine.

### Approaches
We mainly have two levels of parallelism:
* one over features, which parallelizes the procedure of finding the best feature to split at each node;

* one over data, which partitions the training set and parallelizes gradient/Hessian computation as well as the bin sort using local histograms.

### GPU acceleration
To enable GPU, make sure we have supported CUDA devices and CUDA toolkit installed. Then we can compile the code with `-DCUDA=on` flag in cmake. The `-use_gpu` option for `train` is enabled in default. Set `-use_gpu 0` if you want to use CPU-only training. The most computationally intensive part in decision tree building is to sort discretized feature values into bins. If GPU learning is enabled, computation for dense features (whose density exceeds the `data_sparsity_threshold`) will be moved to a single GPU while computation for sparse features takes place on CPU at the same time. On GPU, the training set is partitioned. Each thread will handle a subset of data and construct local feature histograms which will be merged at the end.


## Reference Papers
* Li, Ping. "ABC-Boost: Adaptive Base Class Boost for Multi-Class Classification." _ICML_ 2009.
* Li, Ping. "Robust LogitBoost and Adaptive Base Class (ABC) LogitBoost." _UAI_ 2010.
* Li, Ping and Zhao, Weijie. "Fast ABC-Boost: A Unified Framework for Selecting the Base Class in Multi-Class Classification." _arXiv preprint arXiv:2205.10927_ 2022.
## Copyright and License
ABCBoost is provided under the Apache-2.0 license.
