[English](./README.md) | 中文文档

# ABCBoost

[Fast ABCBoost](https://arxiv.org/pdf/2205.10927.pdf) (Fast Adaptive Base Class Boost)

## 快速上手
### 安装指南
运行一下命令从源代码编译ABCBoost:
```
git clone https://github.com/pltrees/abcboost.git
cd abcboost
mkdir build
cd build
cmake ..
make
cd ..
```
这在`abcboost`目录下创建了三个可执行文件(`abcboost_train`、`abcboost_predict`和`abcboost_clean`)。
`abcboost_train` 是训练模型的可执行文件。
`abcboost_predict` 是使用训练模型进行测试和预测的可执行文件。
`abcboost_clean` 是清理CSV格式数据的可执行文件。

在默认设置中ABCBoost会被编译为单线程程序。如要编译多线程版本，我们需要 [OpenMP](https://en.wikipedia.org/wiki/OpenMP)，(OpenMP通常随Linux系统中的GCC包中一起提供，无需额外下载)。如下打开多线程选项:
```
cmake -DOMP=ON ..
make clean
make
```
请注意，Mac上默认的g++可能不支持OpenMP。 如果需要安装，在`cmake`之前请先执行 `brew install libomp`。


如果我们设置`-DNATIVE=ON`，那么编译器可以根据本机CPU指令集来更好地优化代码: 
```
cmake -DOMP=ON -DNATIVE=ON .. 
make clean
make
```
我们不建议在Mac系统中启用此选项。


如要编译GPU版本的ABCBoost，请安装[NVIDIA CUDA Toolkits](https://developer.nvidia.com/cuda-downloads)并启用选项`CUDA=ON`:
```
cmake -DOMP=ON -DNATIVE=ON -DCUDA=ON ..
make clean 
make
```


### 数据集

我们在`data/`目录下提供了五个数据集: 

[comp_cpu](http://www.cs.toronto.edu/~delve/data/comp-activ/desc.html) 用于回归模型。我们提供了两种格式，CSV和libsvm: `comp_cpu.train.libsvm`, `comp_cpu.train.csv`, `comp_cpu.test.libsvm`, `comp_cpu.test.csv`。请注意，其他树平台可能不支持CSV格式。 

[ijcnn1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1)用于二分类问题。

[covtype](https://archive.ics.uci.edu/ml/datasets/covertype)用于多分类问题。请注意，ABCBoost包不要求类的标签从`0`开始，而其他平台可能有这样的要求。

[mslr10k](https://www.microsoft.com/en-us/research/project/mslr/)用于排序问题。这里只提供了源数据的一小部分样本。 

[Census-Income (KDD) Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/) 用于说明数据清理。


### Lp回归

尽管在一些数据集上使用p>2的时候可以获得更低的误差，我们默认依然使用L2回归 (p=2)。在提供的`comp_cpu`数据集上训练Lp回归，p=2:  
```
./abcboost_train -method regression -lp 2 -data data/comp_cpu.train.csv -J 20 -v 0.1 -iter 1000 
```
这将在当前目录中生成一个名为`comp_cpu.train.libsvm_regression_J10_v0.1_p2.model`的模型文件。接下来我们来测试这个模型，执行以下命令: 
```
./abcboost_predict -data data/comp_cpu.test.csv -model comp_cpu.train.csv_regression_J20_v0.1_p2.model 
```
它会输出两个文件: (1)  `comp_cpu.test.csv_regression_J20_v0.1_p2.testlog`储存了所有迭代的L1和L2的测试误差; (2) `comp_cpu.test.csv_regression_J20_v0.1_p2.prediction` 储存了所有测试数据点的预测值。

`abcboost_train`和`abcboost_predict`这两个可执行文件支持多个输入文件。例如，我们可以同时使用`comp_cpu.train.csv`和`comp_cpu.test.csv`来训练模型:

```
./abcboost_train -method regression -lp 2 -data data/comp_cpu.train.csv data/comp_cpu.test.csv -J 20 -v 0.1 

```
生成的模型文件只按照第一个文件名命名，在本例中，模型文件名依然是`comp_cpu.train.libsvm_regression_J10_v0.1_p2.model`。

### 二分类问题 (Robust LogitBoost) 

我们支持`Robust LogitBoost`和`MART`。因为`Robust LogitBoost`用了二阶导数来计算增益和树的分裂，所以它通常比`MART`效果更好。
```
./abcboost_train -method robustlogit -data data/ijcnn1.train.csv -J 20 -v 0.1 -iter 1000 
./abcboost_predict -data data/ijcnn1.test.csv -model ijcnn1.train.csv_robustlogit_J20_v0.1.model 
```
用户可以将`robustlogit`替换为`mart`以测试不同算法。


### 多分类问题 (Aaptive Base Class Robust LogitBoost) 

我们支持四种训练方法: `robustlogit`、`abcrobustlogit`、`mart`和`abcmart`. 以下示例将在`covtype`数据集(有7个类别)中使用`abcrobustlogit`。为了找到基类(`base class`)，我们需要指定`-search`参数(对于这个数据集，这个参数应在1到7之间)和`-gap`参数(默认值`5`): 
```
./abcboost_train -method abcrobustlogit -data data/covtype.train.csv -J 20 -v 0.1 -iter 1000 -search 2 -gap 10
./abcboost_predict -data data/covtype.test.csv -model covtype.train.csv_abcrobustlogit2g10_J20_v0.1_w0.model 
```
在本例中，模型文件名中的`_w0`我们默认将`-warmup_iter`参数设置为0。 如果将`-search`参数设为`0`，则采取`穷举`策略(等价于在这个数据集中使用`-search 7`)。我们用这个设计来方便用户，使得用户不必知道数据集中类的个数。 


在实际应用中，测试数据往往没有类标签。以下示例用于预测没有类标签的数据，并且只输出预测的类: 
```
./abcboost_predict -data data/covtype.nolabel.test.csv -no_label 1 -model covtype.train.csv_abcrobustlogit2g10_J20_v0.1_w0.model 
```
预测结果储存在`covtype.nolabel.test.csv_abcrobustlogit2g10_J20_v0.1_w0.prediction`文件中。在许多情况下，实践者往往对预测的每个类的概率更感兴趣:

```
./abcboost_predict -data data/covtype.nolabel.test.csv -no_label 1 -save_prob 1 -model covtype.train.csv_abcrobustlogit2g10_J20_v0.1_w0.model 
```
以上代码多输出了一个文件`covtype.nolabel.test.csv_abcrobustlogit2g10_J20_v0.1_w0.probability`。



### 排序问题 (LambdaRank) 

我们可以使用`-method lambdarank`来训练排序模型。请注意，需要指定query/group文件(query文件告诉我们每个query包含多少实例):
```
./abcboost_train -method lambdarank -data data/mslr10k.train -query data/mslr10k.train.query -J 20 -v 0.1 -iter 100 
./abcboost_predict -data data/mslr10k.test -query data/mslr10k.test.query -model mslr10k.train_lambdarank_J20_v0.1.model
```

### 特征装箱 (Histograms) (`-data_max_n_bins`)


在训练之前，我们将每个特征预处理为`0`到`MaxBin-1`之间的整数，其中我们通过`-data_max_n_bins MaxBin`来设置这个上限。更小的`MaxBin`会使训练更快，但是如果`MaxBin`设置得太小，则可能会影响精度。`-data_max_n_bins`的默认值为1000。以下示例我们将此参数更改为500:
```
./abcboost_train -method robustlogit -data data/ijcnn1.train.csv -J 20 -v 0.1 -iter 1000  -data_max_n_bins 500`

```

### GPU 

如果要使用GPU来训练模型，我们可以从命令行来指定GPU设备:
```
CUDA_VISIBLE_DEVICES=0 ./abcboost_train -method robustlogit -data data/ijcnn1.train.csv -J 20 -v 0.1 -iter 1000
```
这里我们指定了`GPU 0`作为训练设备。(使用`nvidia-smi`命令可以查询可用的GPU)

当编译了支持GPU的可执行文件时，如果用户想只使用CPU来进行训练，只需加上选项`-use_gpu 0`。 


### 参数

这里我们说明一些常见参数，并提供了一些示例:
* `-iter` 迭代次数 (默认值 1000)
* `-J` 每棵树中的叶子数 (默认值 20)
* `-v` 学习率 (默认值 0.1)
* `-search` 搜索基类时搜索类的数量 (默认值 2: 我们根据训练损失贪心选择基类). 例如，2表示我们尝试将损失最大的类和损失第二大的类作为基类，并选择损失较低的类作为当前迭代的基类。
* `-n_threads` 线程数 (默认值 1) <strong>仅在启用多线程编译的可执行文件中使用。(在编译前打开cmake的`-DOMP=ON`选项。)</strong>
* `-additional_files` 使用训练数据外的其他文件进行装箱量化。文件名用空格或`,`分隔，例如，`-additional_files file1 file2 file3`。
* `-additional_files_no_label` 使用训练数据外的其他不包含类标签的文件进行装箱量化。文件名用空格或`,`分隔，例如，`-additional_files_no_label file1 file2 file3`。

以2000次迭代、每棵树16个叶子和0.08的学习率训练模型:
```
./abcboost_train -method abcrobustlogit -data data/covtype.train.csv -J 16 -v 0.08 -iter 2000
```

以2000次迭代、每棵树16个叶子和0.08的学习率训练模型，并且启用穷举基类搜索:
```
./abcboost_train -method abcrobustlogit -data data/covtype.train.csv -J 16 -v 0.08 -iter 2000 -search 0 
```
注意，穷举搜索通常生成泛化性能很好的模型，但是同时也需要更多的训练时间。对于`covtype`数据集(其有7个类)，使用`-search 0`等价于使用`-search 7`。

`-additional_files`中提供的额外文件的标签不会被用于训练，它们仅被用于生成(可能)更好的装箱量化。当使用额外文件时，可能可以获得更好的测试结果。


## 更多配置选项:
#### 数据相关:
* `-data_use_mean_as_missing` 用均值来替代缺失的数据
* `-data_min_bin_size` 装箱量化中bin的最小大小
* `-data_sparsity_threshold` 数据稀疏度阈值
* `-data_max_n_bins` 最大bin数量 (默认值 1000)
* `-data_path, -data` 训练和测试数据的路径。我们可以在`-data`中指定多个文件，文件名用空格或`,`分隔。例如: `-data file1 file2 file3`
#### 树相关:
* `-tree_clip_value` 梯度裁剪值 (默认值 50)
* `-tree_damping_factor`, 分母正则化参数 (默认值 1e-100)
* `-tree_max_n_leaves`, -J 树的最大叶子数 (默认值 20)
* `-tree_min_node_size` 树的叶子节点最小包含实例个数 (默认值 10)
#### 模型相关:
* `-model_use_logit`, 是否使用二阶导数
* `-model_data_sample_rate` 训练时的实例采样率(默认值 1.0)
* `-model_feature_sample_rate` 训练时的特征采样率 (默认值 1.0)
* `-model_shrinkage`, `-shrinkage`, `-v`, 学习率 (默认值 0.1)
* `-model_n_iterations`, `-iter` 迭代数 (默认值 1000)
* `-model_save_every`, `-save` 每多少次迭代保存模型 (默认值 100)
* `-model_eval_every`, `-eval` 每多少次迭代计算损失函数 (默认值 1)
* `-model_name`, `-method` 训练方法，可选参数有regression/lambdarank/mart/abcmart/robustlogit/abcrobustlogit (默认值 robustlogit)
* `-model_pretrained_path`, `-model` 模型路径
#### Adaptive Base Class (ABC) 相关:
* `-model_base_candidate_size`, `base_candidates_size`, `-search` (默认值 2) 在abcmart和abcrobustlogit训练方法中的基类搜索大小
* `-model_gap`, `-gap` (默认值 5) 两次基类搜索之间的间隔。例如，`-model_gap 2`表示我们会在第1, 4, 6, ...次迭代中进行基类搜索。
* `-model_warmup_iter`, `-warmup_iter` (默认值 0) 在ABC方法中使用warmup的迭代次数。当我们的基类搜索参数很小的时候(`-search`)，他可能对包含大量类的数据集有帮助。
* `-model_warmup_use_logit`, `-warmup_use_logit` 0/1 (默认值 1) 是否在warmup阶段使用二阶导数。
* `-model_abc_sample_rate`, `-abc_sample_rate` (默认值 1.0) 用于基类搜索中的采样率。
* `-model_abc_sample_min_data` `-abc_sample_min_data` (默认值 2000) 在基类搜索中的最小采样数量。这个参数仅在`-abc_sample_rate`小于`1`时生效。

#### 回归相关:
* `-regression_lp_loss`, `-lp` (默认值 2.0) 是否使用Lp范数来取代L2范数。须指定p，(p >= 1.0)。
* `-regression_test_lp`, `-test_lp` (默认 空) 将Lp范数作为测试日志中的附加列。须指定p， (p 。
* `-regression_use_hessian` 0/1 (默认值 1) 是否在回归模型中使用二阶导数。这个参数仅在设置了`-regression_lp_loss p`并且`p`大于`2`时有效。
* `-regression_huber_loss`, `-huber` 0/1 (默认值 0) 是否在回归模型中使用Huber损失
* `-regression_huber_delta`, `-huber_delta` Huber损失中的delta参数。这个参数仅在`-regression_huber_loss 1`被设置时生效。
#### 并行:
* `-n_threads`, `-threads` (默认值 1) 线程数
* `-use_gpu` 0/1 (如果使用CUDA编译，则默认值为 1) 是否使用GPU来训练模型。此参数仅在可执行文件用CUDA编译时生效(即，在`cmake`中设置了选项`-DCUDA=on`)。
#### 其他:
* `-save_log`, 0/1 (默认值 0) 是否保存日志文件
* `-save_model`, 0/1 (默认值 1) 是否储存模型
* `-save_prob`, 0/1 (默认值 0) 是否保存分类预测概率
* `-save_importance`, 0/1 (默认值 0) 是否保存训练特征重要度
* `-no_label`, 0/1 (默认值 0) 测试文件是否没有标签。仅当测试数据在测试中没有标签时，才可启用该选项以输出预测文件。
* `-test_auc`, 0/1 (默认值 0) 是否在测试中计算AUC
* `-stop_tolerance` (默认值 2e-14) 仅适用于非回归模型，例如分类模型。当总损失小于`-stop_tolerance`时，训练将会停止。
* `-regression_stop_factor` (默认值 1e-6) 回归模型的自动停止标准和分类模型不同，因为回归模型的目标规模是未知的，我们自适应地将回归停止阈值设置为`regression_stop_factor * total_loss / sum(y^p)`，其中`y`是回归模型的目标，`p`是`-regression_lp_loss`中指定的值。
* `-regression_auto_clip_value` 0/1 (默认值 1) 是否在回归模型中使用自适应裁剪值来计算叶子节点上的预测值。启用时，自适应裁剪值的计算为`tree_clip_value * max_y - min_y`，其中`tree_clip_value`通过`-tree_clip_value`设置，`max_y`和`min_y`分别是回归目标的最大和最小值。
* `-gbrank_tau` (默认值 0.1) gbrank的tau参数。

## R语言支持
我们提供了一个R语言库来帮助用户从R语言中调用ABCBoost的函数。
要打包和安装这个库，请在`abcboost/`目录中键入以下指令:
```
cd ..
R CMD build abcboost
```

为了方便用户，我们还在`R/`目录中提供了预打包的`abcboost_1.0.0.tar.gz`和`abcboost_1.0.0_mult.tar.gz`，分别用于单线程版本和多线程版本。要安装(单线程)软件包，请在R console中键入
```
install.packages('R/abcboost_1.0.0.tar.gz', repos = NULL, type = 'source')
```
可以使用`setwd`来更改当前的工作目录。请注意，如果希望将单线程版本替换为多线程版本，我们应该先删除已经安装的包(`remove.packages('abcboost')`)。


函数说明(无需复制到R console):
```
# No need to copy to console
# abcboost_train: (train_Y,train_X,model_name,iter,leaves,shinkage,params=NULL)
# abcboost_test: (test_Y,test_X,model,params=NULL)
# abcboost_predict: (test_X,model,params=NULL)
# abcboost_save_model: function(model,path)
# abcboost_load_model: function(path)
```
这里我们展示了一个训练和测试的例子:
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


## Matlab支持
我们提供了一个Matlab接口来帮助用户从Matlab调用ABCBoost的函数。
要在Matlab中编译Matlab mex文件，请执行以下操作:
```
cd src/matlab
compile    % single-thread version 
```
或者
```
cd src/matlab
compile_mult_thread 
```

可以在Matlab console中使用`mex -setup cpp`来指定C++编译器。 


为了方便用户，我们分别在`matlab/linux`和`matlab/windows`中为Linux和Windows用户提供了编译后的可执行文件。



假设可执行文件储存在`matlab/linux`或`matlab/windows`中，我们这里展示了一个训练和测试的例子:
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

## Python支持
我们通过`pybind11`提供了python支持。
在编译前，请安装`pybind11`:

`python3 -m pip install pybind11`

在Linux(非 Mac)上编译单线程版本:
```
cd python/linux
bash compile_py.sh
```
编译完成后，将产生一个shared library `abcboost.so`。

类似地，Mac也有两个文件夹`python/mac_m1`和`python/mac_x86`。

对于Windows，我们在`python\windows`目录下提供了shared (python3.10) library `abcboost.pyd`。 

确保`abcboost.so`(在Windows下为`abcboost.pyd`)在当前目录中。
将以下代码粘贴到`python3` shell中:
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

## 数据清理和类别特征(Categorical Feature)的处理

我们提供了一个可执行文件`abcboost_clean`用于清理CSV文件和检测类别特征。类别特征将被编码为one-hot表示，并被放在数字特征之后。处理后的数据集(默认情况下)将以libsvm格式储存。用户也可以选择指定输出成CSV格式。

我们来看一个示例: [Census-Income (KDD) Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/). 首先，这是一个二分类数据集合，标签在最后一列，为`- 50000`和`50000+`。这两个标签会被分别转换为`0`和`1`，并被放置在输出数据的第一列中。此数据集包含缺失值和许多由字符串表示的分类特征。有趣的是，在这个数据集`Census-Income (KDD) Data Set`中，我们注意到`MART`略优于`Robust LogitBoost`。


执行以下命令将为训练数据`census-income.data`和测试数据`census-income.test`生成清理后的`csv`文件:
```
./abcboost_clean -data data/census-income.data data/census-income.test -label_column -1 -cleaned_format csv  
```
`-label_column -1`表示`最后`一列为标签。如果我们希望以不同格式储存数据，可以将`csv`替换为`libsvm`。默认选择为`libsvm`。以下是上述命令输出的结尾: 

```
Found non-numeric labels:
(50000+.)->1 (- 50000.)->0
Cleaning summary: | # data: 299285 | # numeric features 14 | # categorical features: 28 | # converted features: 401 | # classes: 2
```

在上面的命令中，我们在`-data`中指定了多个文件。`census-income.data`和`census-income.test`将在内部被合并成一个文件以处理类别特征。或者我们也可以先清理`census-income.data`，然后再使用储存在`census-income.data.cleaninfo`的信息来单独处理`census-income.test`: 

```
./abcboost_clean -data data/census-income.data -label_column -1
./abcboost_clean -data data/census-income.test -cleaninfo data/census-income.data.cleaninfo -cleaned_format csv 
```
请注意，因为额外文件中可能包含额外的类别特征，上述的两种方法可能产生不同的输出文件。


总之，`abcboost_clean`具有许多功能，接下来我们列出了诸多选项和其对应的解释: 


* `-data` 需要清理的文件。我们通过在`-data`中指定多个文件名来同时清理训练、测试和验证数据集（文件名之间用半角逗号隔开）。
* `-ignore_columns` CSV文件中需要忽略的列。我们可以用(半角)逗号来分隔多个列的下标，例如，`-ignore_columns 1,3,-2`忽略了第一列，第三列，和倒数第二列。下标是从1开始计数的。逗号和列下标之间不应有空格。
* `-ignore_rows` CSV文件中需要忽略的行。我们可以用(半角)逗号来分隔多个行的下标。
* `-label_column` (默认值 1) 包含标签的列下标
* `-category_limit` (默认值 10000) 一个类别特征中最多可以出现的类别数。在自动检测中，如果在一个特征中出现的类别特征数超过了这个限制，我们会将该列视为数字特征，并将非数字值视为缺失值。我们可以通过指定`-additional_categorical_columns`来绕过这个限制。
* `-additional_categorical_columns` 指定额外的类别特征列(它将覆盖自动检测的结果)
* `-additional_numeric_columns` 指定额外数字特征。这些列中的非数字值都将被视为确实值。
* `-missing_values` (默认值 ne,na,nan,none,null,unknown,,?) 指定缺失值的表示方式(不区分大小写)。
* `-missing_substitution` (默认值 0) 我们会将所有缺失值替换为此数字
* `-cleaned_format` (默认值 libsvm) 清理后的数据集输出格式。可以将其指定为csv或libsvm。我们建议使用libsvm格式，libsvm格式对于one-hot编码有更紧凑的表示。
* `-cleaninfo` 指定`.cleaninfo`文件。如果这个文件没被指定，我们会生成一个后缀为`.cleaninfo`的文件，此文件包含了数据清理的信息，例如标签列、类别特征的映射等。通过指定`-cleaninfo`，我们可以使用与上次清理相同的映射来清理其他数据。例如，我们首先清理训练数据，然后使用训练数据的`.cleaninfo`文件来清理测试数据以确保它们具有相同的特征映射。请注意`-ignore_rows`的值未被保存在`.cleaninfo`中。

## 参考文献
* Ping Li. [ABC-Boost: Adaptive Base Class Boost for Multi-Class Classification](https://icml.cc/Conferences/2009/papers/417.pdf). ICML 2009.
* Ping Li. [Robust LogitBoost and Adaptive Base Class (ABC) LogitBoost](https://event.cwi.nl/uai2010/papers/UAI2010_0282.pdf). UAI 2010.
* Ping Li and Weijie Zhao. [Fast ABC-Boost: A Unified Framework for Selecting the Base Class in Multi-Class Classification](https://arxiv.org/pdf/2205.10927.pdf).  arXiv:2205.10927 2022.
* Ping Li and Weijie Zhao. [Package for Fast ABC-Boost](https://arxiv.org/pdf/2207.08770.pdf).  arXiv:2207.08770, 2022.
* Ping Li and Weijie Zhao. [pGMM Kernel Regression and Comparisons with Boosted Trees](https://arxiv.org/pdf/2207.08667.pdf).   arXiv:2207.08667, 2022.
* Lecture notes on trees & boosting (pages 14-77) [www.stat.rutgers.edu/home/pingli/doc/PingLiTutorial.pdf](www.stat.rutgers.edu/home/pingli/doc/PingLiTutorial.pdf)


## 版权和许可
ABCBoost根据Apache-2.0许可证提供。
