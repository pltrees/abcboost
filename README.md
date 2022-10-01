English | [中文文档](./README_Chinese.md)

# ABCBoost 

This toolkit consists of ABCBoost, the implementation of [Fast ABCBoost](https://arxiv.org/pdf/2205.10927.pdf) (Fast Adaptive Base Class Boost). 

## Quick Start
### Installation guide
Run the following commands to build ABCBoost from source:
```
git clone https://github.com/pltrees/abcboost.git
cd abcboost
mkdir build
cd build
cmake ..
make
cd ..
```
This will create three executables (`abcboost_train`, `abcboost_predict`, and `abcboost_clean`) in the `abcboost` directory.
`abcboost_train` is the executable to train models.
`abcboost_predict` is the executable to validate and inference using trained models.
`abcboost_clean` is the executable to clean csv data.

The default setting builds ABCBoost as a single-thread program.  To build ABCBoost with multi-thread support [OpenMP](https://en.wikipedia.org/wiki/OpenMP) (OpenMP comes with the system GCC toolchains on Linux), turn on the multi-thread option:
```
cmake -DOMP=ON ..
make clean
make
```
Note that the default g++ on Mac may not support OpenMP.  To install, execute `brew install libomp` before `cmake`.


If we set `-DNATIVE=ON`, the compiler may better optimize the code according to specific native CPU instructions: 
```
cmake -DOMP=ON -DNATIVE=ON .. 
make clean
make
```
We do not recommend to turn on this option on Mac. 


To build ABCBoost with GPU support, install [NVIDIA CUDA Toolkits](https://developer.nvidia.com/cuda-downloads) and set the option `CUDA=ON`:
```
cmake -DOMP=ON -DNATIVE=ON -DCUDA=ON ..
make clean 
make
```


### Datasets 

Five datasets are provided under `data/` folder: 

[comp_cpu](http://www.cs.toronto.edu/~delve/data/comp-activ/desc.html) for regression, in both CSV  and libsvm formats: `comp_cpu.train.libsvm`, `comp_cpu.train.csv`, `comp_cpu.test.libsvm`, `comp_cpu.test.csv`. Note that other tree platforms may not support the CSV format. 

[ijcnn1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1) for binary classification. 

[covtype](https://archive.ics.uci.edu/ml/datasets/covertype) for multi-class classification. Note that ABCBoost package does not require class labels to start from `0` while other platforms may require so. 

[mslr10k](https://www.microsoft.com/en-us/research/project/mslr/) for ranking. Only a small subset is provided here. 


[Census-Income (KDD) Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/) to illustrate data cleaning and categorical feature processing. 


### Lp Regression 

L2 regression (p=2) is the default, although in some datasets using p>2 achieves lower errors.  To train Lp regression on the provided `comp_cpu` dataset, with p=2:  
```
./abcboost_train -method regression -lp 2 -data data/comp_cpu.train.csv -J 20 -v 0.1 -iter 1000 
```
This will generate a model named `comp_cpu.train.libsvm_regression_J10_v0.1_p2.model` on the current directory. For testing, execute 
```
./abcboost_predict -data data/comp_cpu.test.csv -model comp_cpu.train.csv_regression_J20_v0.1_p2.model 
```
which outputs two files: (1)  `comp_cpu.test.csv_regression_J20_v0.1_p2.testlog` which stores the history of test L1 and L2 loss for all the iterations; and (2) `comp_cpu.test.csv_regression_J20_v0.1_p2.prediction` which stores the predictions for all testing data points.  

The train/predict executables support multiple input data files. For example, we can use both `comp_cpu.train.csv` and `comp_cpu.test.csv` for training the model:

```
./abcboost_train -method regression -lp 2 -data data/comp_cpu.train.csv data/comp_cpu.test.csv -J 20 -v 0.1 

```
Note that the model is named accordingly only the first input data file, in this case, the model name is still `comp_cpu.train.libsvm_regression_J10_v0.1_p2.model`. 


### Binary Classification (Robust LogitBoost) 

We support both `Robust LogitBoost` and `MART`. Because `Robust LogitBoost` uses second-order information to compute the gain for tree plits, it often improves `MART`. 
```
./abcboost_train -method robustlogit -data data/ijcnn1.train.csv -J 20 -v 0.1 -iter 1000 
./abcboost_predict -data data/ijcnn1.test.csv -model ijcnn1.train.csv_robustlogit_J20_v0.1.model 
```
Users can replace `robustlogit` by `mart` to test different algorithms. 


### Multi-Class Classification (Aaptive Base Class Robust LogitBoost) 

We support four training methods: `robustlogit`,  `abcrobustlogit`, `mart`, and `abcmart`. The following example is for `abcrobustlogit` on `covtype` dataset which has `7` classes. In order to identify the `base class`, we need to specify the `-search` parameter (between 1 and 7 for this dataset) and `-gap` parameter (`5` by default): 
```
./abcboost_train -method abcrobustlogit -data data/covtype.train.csv -J 20 -v 0.1 -iter 1000 -search 2 -gap 10
./abcboost_predict -data data/covtype.test.csv -model covtype.train.csv_abcrobustlogit2g10_J20_v0.1_w0.model 
```
In this example, the `_w0` in the model name means we set `-warmup_iter` to be 0 by default. Note that if the `-search` parameter is set to be `0`, then the `exhaustive` strategy is adopted (which is the same as `-search 7` in this example). We choose this design convention so that readers do not have to know the number of classes of the dataset. 


In practice, the test data would not have labels. The following example outputs only the predicted class labels: 
```
./abcboost_predict -data data/covtype.nolabel.test.csv -no_label 1 -model covtype.train.csv_abcrobustlogit2g10_J20_v0.1_w0.model 
```
in `covtype.nolabel.test.csv_abcrobustlogit2g10_J20_v0.1_w0.prediction` file. In many scenarios, practitioners are often more interested in the predicted class probabilities. The next example

```
./abcboost_predict -data data/covtype.nolabel.test.csv -no_label 1 -save_prob 1 -model covtype.train.csv_abcrobustlogit2g10_J20_v0.1_w0.model 
```
outputs an additional file `covtype.nolabel.test.csv_abcrobustlogit2g10_J20_v0.1_w0.probability`. 


### Ranking (LambdaRank) 

Ranking tasks are supported by using `-method lambdarank`. Note that the query/group file need to be specified (the query file tells us how many instances in the data for each query):
```
./abcboost_train -method lambdarank -data data/mslr10k.train -query data/mslr10k.train.query -J 20 -v 0.1 -iter 100 
./abcboost_predict -data data/mslr10k.test -query data/mslr10k.test.query -model mslr10k.train_lambdarank_J20_v0.1.model
```




### Feature Binning (Histograms) (`-data_max_n_bins`)


Before the training stage, each feature is preprocessed to be integers between `0` and `MaxBin-1` where we set the up limit by `-data_max_n_bins MaxBin`. Smaller `MaxBin` results in faster training, but it may hurt the accuracy if `MaxBin` is set to be too small. The default value of `-data_max_n_bins` is 1000. The following example changes this parameter to 500:
```
./abcboost_train -method robustlogit -data data/ijcnn1.train.csv -J 20 -v 0.1 -iter 1000  -data_max_n_bins 500`

```

### GPU 

If the executables are compiled with GPU support, we can specify the GPU device from the command line:
```
CUDA_VISIBLE_DEVICES=0 ./abcboost_train -method robustlogit -data data/ijcnn1.train.csv -J 20 -v 0.1 -iter 1000
```
Here we specify `GPU 0` as the device. (Use `nvidia-smi` to find out available GPUs)

If users hope to use GPU-complied executables for CPU-only, simply add `-use_gpu 0`. 


### Parameters

Here we illustrate some common parameters and provide some examples:
* `-iter` number of iterations (default 1000)
* `-J` number of leaves in a tree (default 20)
* `-v` learning rate (default 0.1)
* `-search` searching size for the base class (default 2: we greedily choose the base classes according to the training loss). For example, 2 means we try the class with the greatest loss and the class with the second greatest loss as base class and pick the one with lower loss as the base class for the current iteration.
* `-n_threads` number of threads (default 1) <strong>It can only be used when multi-thread is enabled. (Compile the code with `-DOMP=ON` in cmake.)</strong>
* `-additional_files` using other files to do bin quantization besides the training data. File names are separated by `,`, e.g., `-additional_files file1 file2 file3`.
* `-additional_files_no_label` using other unlabeled files to do bin quantization besides the training data. File names are separated by space or `,`, e.g., `-additional_files_no_label file1 file2 file3`.

To train the model with 2000 iterations, 16 leaves per tree and 0.08 learning rate:
```
./abcboost_train -method abcrobustlogit -data data/covtype.train.csv -J 16 -v 0.08 -iter 2000
```

To train the model with 2000 iterations, 16 leaves per tree, 0.08 learning rate and enable the exhaustive base class searching:
```
./abcboost_train -method abcrobustlogit -data data/covtype.train.csv -J 16 -v 0.08 -iter 2000 -search 0 
```
Note that the exhaustive searching produces better-generalized model while requiring substantially more time. For the `covtype` dataset (which has 7 classes), using `-search 0` is effectively equivalent to `-search 7`. 

The labels in the specified additional files are not used in the training. Only the feature values are used to generate (potentially) better quantization. Better testing results may be obtained when using additional files


## More Configuration Options:
#### Data related:
* `-data_use_mean_as_missing`
* `-data_min_bin_size` minimum size of the bin
* `-data_sparsity_threshold`
* `-data_max_n_bins` max number of bins (default 1000)
* `-data_path, -data` path to train/test data. We can specify multiple data in `-data`. The file names are separated by space or comma. For example, `-data file1 file2 file3`
#### Tree related:
* `-tree_clip_value` gradient clip (default 50)
* `-tree_damping_factor`, regularization on denominator (default 1e-100)
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
* `-model_name`, `-method` regression/lambdarank/mart/abcmart/robustlogit/abcrobustlogit (default robustlogit)
* `-model_pretrained_path`, `-model`
#### Adaptive Base Class (ABC) related:
* `-model_base_candidate_size`, `base_candidates_size`, `-search` (default 2) base class searching size in abcmart/abcrobustlogit
* `-model_gap`, `-gap` (default 5) the gap between two base class searchings. For example, `-model_gap 2` means we will do the base class searching in iteration 1, 4, 6, ...
* `-model_warmup_iter`, `-warmup_iter` (default 0) the number of iterations that use normal boosting before ABC method kicks in. It might be helpful for datasets with a large number of classes when we only have a limited base class searching parameter (`-search`) 
* `-model_warmup_use_logit`, `-warmup_use_logit` 0/1 (default 1) whether use logitboost in warmup iterations.
* `-model_abc_sample_rate`, `-abc_sample_rate` (default 1.0) the sample rate used for the base class searching
* `-model_abc_sample_min_data` `-abc_sample_min_data` (default 2000) the minimum sampled data for base class selection. This parameter only takes into effect when `-abc_sample_rate` is less than `1.0`
#### Regression related:
* `-regression_lp_loss`, `-lp` (default 2.0) whether use Lp norm instead of L2 norm. p (p >= 1.0) has to be specified
* `-regression_test_lp`, `-test_lp` (default none) display Lp norm as an additional column in test log. p (p >= 1.0) has to be specified
* `-regression_use_hessian` 0/1 (default 1) whether use second-order derivatives in the regression. This parameter only takes into effect when `-regression_lp_loss p` is set and `p` is greater than `2`.
* `-regression_huber_loss`, `-huber` 0/1 (default 0) whether use huber loss
* `-regression_huber_delta`, `-huber_delta` the delta parameter for huber loss. This parameter only takes into effect when `-regression_huber_loss 1` is set
#### Parallelism:
* `-n_threads`, `-threads` (default 1)
* `-use_gpu` 0/1 (default 1 if compiled with CUDA) whether use GPU to train models. This parameter only takes into effect when the executable is complied with CUDA (i.e., the flag `-DCUDA=on` is enabled in `cmake`).
#### Other:
* `-save_log`, 0/1 (default 0) whether save the runtime log to file
* `-save_model`, 0/1 (default 1)
* `-save_prob`, 0/1 (default 0) whether save the prediction probability for classification tasks
* `-save_importance`, 0/1 (default 0) whether save the feature importance in the training
* `-no_label`, 0/1 (default 0) It should only be enabled to output prediction file when the testing data has no label in test
* `-test_auc`, 0/1 (default 0) whether compute AUC in test
* `-stop_tolerance` (default 2e-14) It works for all non-regression tasks, e.g., classification. The training will stop when the total training loss is less than the stop tolerance.
* `-regression_stop_factor` (default 1e-6) The auto stopping criterion is different from the classification task because the scale of the regression target is unknown. We adaptively set the regression stop tolerate to `regression_stop_factor * total_loss / sum(y^p)`, where `y` is the regression targets and `p` is the value specified in `-regression_lp_loss`.
* `-regression_auto_clip_value` 0/1 (default 1) whether use our adaptive clipping value computation for the predict value on terminal nodes. When enabled, the adaptive clipping value is computed as `tree_clip_value * max_y - min_y` where `tree_clip_value` is set via `-tree_clip_value`, `max_y` and `min_y` are the maximum and minimum regression target value, respectively.
* `-gbrank_tau` (default 0.1) The tau parameter for gbrank.

## R Support
We provide an R library to enable calling ABCBoost subroutines from R.
To build and install the library, type the following command in `abcboost/`:
```
cd ..
R CMD build abcboost
```

For users' convience, we also provide, under `R/`,  the pre-built `abcboost_1.0.0.tar.gz` and `abcboost_1.0.0_mult.tar.gz`, for single-thread version and multi-thread version, respectively. To install the (single-thread) package, in R console, type 
```
install.packages('R/abcboost_1.0.0.tar.gz', repos = NULL, type = 'source')
```
One can use `setwd` to change the current working directory. Note that we should first remove the package (`remove.packages('abcboost')`) if we hope to replace the single-thread version with the multi-thread version. 


Function description (no need to copy to console):
```
# No need to copy to console
# abcboost_train: (train_Y,train_X,model_name,iter,leaves,shinkage,params=NULL)
# abcboost_test: (test_Y,test_X,model,params=NULL)
# abcboost_predict: (test_X,model,params=NULL)
# abcboost_save_model: function(model,path)
# abcboost_load_model: function(path)
```
Here we show an example of training and testing:
```
library(abcboost)
data <- read.csv(file='data/covtype.train.csv',header=FALSE)
X <- data[,-1]
Y <- data[,1]
data <- read.csv(file='data/covtype.test.csv',header=FALSE)
testX <- data[,-1]
testY <- data[,1]
# The last argument of abcboost_train is optional. 
# We use n_threads as an example
# All command line supported parameters can be passed via list: 
# list(parameter1=value1, parameter2=value2,...)
model <- abcboost_train(Y,X,"abcrobustlogit",100,20,0.1,list(n_threads=1,search=2,gap=5))
# abcboost_save_model(model,'mymodel.model')
# model <- abcboost_load_model('mymodel.model')
res <- abcboost_test(testY,testX,model,list(test_auc=1))
# predict without label 
res <- abcboost_predict(testX,model,list(test_auc=1))
# We also provide a method to read libsvm format data into sparse array
data <- abcboost_read_libsvm('data/covtype.train.libsvm')
X <- data$X
Y <- data$Y
data <- abcboost_read_libsvm('data/covtype.test.libsvm')
testX <- data$X
testY <- data$Y
# X can be a either a dense matrix or a sparse matrix
# The interface is the same as the dense case, 
# but with better performance for sparse data
model <- abcboost_train(Y,X,"abcrobustlogit",100,20,0.1,list(n_threads=1,search=2,gap=5))
res <- abcboost_test(testY,testX,model)
```


## Matlab Support
We provide a Matlab wrapper to call ABCBoost subroutines from Matlab.
To compile the Matlab mex files in Matlab:
```
cd src/matlab
compile    % single-thread version 
```
or 
```
cd src/matlab
compile_mult_thread 
```

One can use `mex -setup cpp` in a matlab console to check the specified C++ compiler. 


For the convenience of users, we provide the compiled executables for both Linux and Windows under `matlab/linux` and `matlab/windows` respectively. 



Assume the executables are stored in `matlab/linux` or `matlab/windows`. Here we show an example of training and testing:
```
tr = load('../../data/covtype.train.csv');
te = load('../../data/covtype.test.csv');
Y = tr(:,1);
X = tr(:,2:end);
testY = te(:,1);
testX = te(:,2:end);

params = struct;
params.n_threads = 1;
params.search = 2;
params.gap = 5;
% The params argument is optional. 
% We use n_threads as an example
% All command line supported parameters can be passed via params: params.parameter_name = value
model = abcboost_train(Y,X,'abcrobustlogit',100,20,0.1,params);
% abcboost_save(model,'mymodel.model');
% model = abcboost_load('mymodel.model');
params.test_auc = 1;
res = abcboost_test(testY,testX,model,params);
% predict without label 
res = abcboost_predict(testX,model,params);

% Sparse matlab matrix is also supported
% For example, we included the libsvmread.c from the LIBSVM package for data loading
[Y, X] = libsvmread('../../data/covtype.train.libsvm');
[testY, testX] = libsvmread('../../data/covtype.test.libsvm');
% Here X and testX are sparse matrices
model = abcboost_train(Y,X,'abcrobustlogit',100,20,0.1,params);
res = abcboost_test(testY,testX,model);
```

## Python Support
We provide the python support through `pybind11`.
Before the compilation, `pybind11` should be installed:

`python3 -m pip install pybind11`

To compile the single-thread version on Linux (not Mac):
```
cd python/linux
bash compile_py.sh
```
After the compilation, a shared library `abcboost.so` is generated. 

Analogously, there are two folders for Mac: `python/mac_m1`, `python/mac_x86`.

For windows, we provide the shared (python3.10) library `abcboost.pyd` under `python/windows`. 

Make sure `abcboost.so` (or `abcboost.pyd` for Windows) is in the current directory.
Paste the following code in a `python3` interactive shell:
```
import numpy as np
import abcboost
# We use a matrix-format sample data here
data = np.genfromtxt('../../data/covtype.train.csv',delimiter=',').astype(float)
#
Y = data[:,0]
X = data[:,1:]
data = np.genfromtxt('../../data/covtype.test.csv',delimiter=',').astype(float)
testY = data[:,0]
testX = data[:,1:]
model = abcboost.train(Y,X,'abcrobustlogit',100,20,0.1)
# All command line supported parameters can be passed as optional keyword arguments
# For example:
# model = abcboost.train(Y,X,'abcrobustlogit',100,20,0.1,search=2,gap=5,n_threads=1)
# abcboost.save(model,'mymodel.model')
# model = abcboost.load('mymodel.model')
res = abcboost.test(testY,testX,model,test_auc=1)
# predict without label 
res = abcboost.predict(testX,model,test_auc=1)
# Alternatively, we also support libsvm-format sparse matrix
# We use sklearn to load libsvm format data as a scipy.sparse matrix
# sklearn can be installed as: python3 -m pip install scikit-learn
import sklearn
import sklearn.datasets
# X is a scipy.sparse matrix
[X, Y] = sklearn.datasets.load_svmlight_file('../../data/covtype.train.libsvm')
[testX, testY] = sklearn.datasets.load_svmlight_file('../../data/covtype.train.libsvm')
# The training and testing interfaces are unified for both dense and sparse matrices
model = abcboost.train(Y,X,'abcrobustlogit',100,20,0.1)
res = abcboost.test(testY,testX,model)
```

## Data Cleaning and Categorical Feature Processing

We provide an executable `abcboost_clean` for cleaning CSV files and detecting categorical feature. The categorical features will be encoded into one-hot representations and placed after the numerical features. The processed dataset will be stored by (by default) in the libsvm format but users also choose to specify the CSV format. 

We provide an illustrative example: [Census-Income (KDD) Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/). First of all, the binary labels are `- 50000`  and  `50000+` in the last column, which will be converted to respectively `0` and `1` and be placed in the first column of the processed dataset. This dataset contains missing values and many categorical features represented by strings. Interestingly, on this dataset `Census-Income (KDD) Data Set`, we notice that `MART` slightly outperforms `Robust LogitBoost`. 


Executing the following terminal command will generate cleaned `csv` files for both training data `census-income.data` and testing data `census-income.test`: 
```
./abcboost_clean -data data/census-income.data data/census-income.test -label_column -1 -cleaned_format csv  
```
`-label_column -1` indicate the `last` column is for the labels.  We can replace `csv` by `libsvm` if we hope to store the data in a different format. The default choice is `libsvm`.  The following is the end of the output of the above command: 

```
Found non-numeric labels:
(50000+.)->1 (- 50000.)->0
Cleaning summary: | # data: 299285 | # numeric features 14 | # categorical features: 28 | # converted features: 401 | # classes: 2
```

In the above command, multiple files are specified in `-data`. The `census-income.data` and `census-income.test` will be internally combined as one file to process categorical features. Alternatively, we can also first clean `census-income.data` and then use the information stored in `census-income.data.cleaninfo` to separately process `census-income.test`: 

```
./abcboost_clean -data data/census-income.data -label_column -1
./abcboost_clean -data data/census-income.test -cleaninfo data/census-income.data.cleaninfo -cleaned_format csv 
```
Note that the above two ways may not necessarily generate the same results because the additional files may contain additional categories. In this particular example, one can check that the results are the same.  

We also provide a normalization parameter `-normalize` in `./abcboost_clean`: `zero_to_one`, `minus_one_to_one`, `gaussian`. For example:
```
./abcboost_clean -data data/census-income.data -label_column -1 -normalize zero_to_one
./abcboost_clean -data data/census-income.test -cleaninfo data/census-income.data.cleaninfo -cleaned_format csv 
```
The `-normalize` parameter will be saved in the `.cleaninfo` file. We do not need to specify it again if `-cleaninfo` is set.

In summary, `abcboost_clean` has a variety of functionalities. In the following, we list the options and explanations. 


* `-data` the data files to clean. We may clean the training, testing, validating dataset together by specifying multiple file names in `-data` (file names are separated by space or comma).
* `-ignore_columns` the columns to ignore in the CSV file. Multiple columns can be separated by commas, e.g., `-ignore_columns 1,3,-2` ignores the first, third, and the second last columns. The index is one-based. There should be no space between the comma and the column indices
* `-ignore_rows` the rows to ignore in the CSV file. Multiple rows can be separated by commas
* `-label_column` (default 1) the column contains the label
* `-category_limit` (default 10000) the limit of the categories in a feature. In the auto detection, we will consider the column as a numeric column and treat non-numeric values as missing if the detected categories in the feature exceed this limit. We can specify the `-additional_categorical_columns` to bypass this limit
* `-additional_categorical_columns` specifies additional categorical columns (it will override the auto categorical feature detection result)
* `-additional_numeric_columns` specifies additional numeric columns (it will override the auto categorical feature detection result). All non-numeric values in those columns will be considered as missing
* `-missing_values` (default ne,na,nan,none,null,unknown,,?) specifies the possible missing values (case-insensitive).
* `-missing_substitution` (default 0) we will substitute all missing values with this specified number
* `-cleaned_format` (default libsvm) the output format of the cleaned data. It can be specified to csv or libsvm. We suggest to use libsvm for a compact representation of the one-hot encoded categorical values.
* `-cleaninfo` specifies the `.cleaninfo` file. If this is unspecified. We will generate a file with a `.cleaninfo` suffix that contains the cleaning information, e.g., label columns, categorical mapping, etc. Specifying `-cleaninfo` enables us to clean other data with the same mapping of the previous cleaning. For example, we clean the training data first. And later we can use the `.cleaninfo` of the training data to clean the testing data to ensure they have the same feature mapping. Note that the `-ignore_rows` is not saved in the `.cleaninfo`.
* `-normalize` (default none) specifies the method to normalize data. Options: `zero_to_one`, `minus_one_to_one`, and `gaussian`.

## References
* Ping Li. [ABC-Boost: Adaptive Base Class Boost for Multi-Class Classification](https://icml.cc/Conferences/2009/papers/417.pdf). ICML 2009.
* Ping Li. [Robust LogitBoost and Adaptive Base Class (ABC) LogitBoost](https://event.cwi.nl/uai2010/papers/UAI2010_0282.pdf). UAI 2010.
* Ping Li and Weijie Zhao. [Fast ABC-Boost: A Unified Framework for Selecting the Base Class in Multi-Class Classification](https://arxiv.org/pdf/2205.10927.pdf).  arXiv:2205.10927 2022.
* Ping Li and Weijie Zhao. [Package for Fast ABC-Boost](https://arxiv.org/pdf/2207.08770.pdf).  arXiv:2207.08770, 2022.
* Ping Li and Weijie Zhao. [pGMM Kernel Regression and Comparisons with Boosted Trees](https://arxiv.org/pdf/2207.08667.pdf).   arXiv:2207.08667, 2022.
* Lecture notes on trees & boosting (pages 14-77) [www.stat.rutgers.edu/home/pingli/doc/PingLiTutorial.pdf](www.stat.rutgers.edu/home/pingli/doc/PingLiTutorial.pdf)


## Copyright and License
ABCBoost is provided under the Apache-2.0 license.
