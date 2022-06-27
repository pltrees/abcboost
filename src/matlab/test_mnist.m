[Y,X] = libsvmread('../../data/mnist.train');
[testY,testX] = libsvmread('../../data/mnist.test');

M = 3;
J = 20;
v = 0.1;
s = 10;
g = 0;

params = struct;
params.n_threads = 1;

model = abcboost_train(Y,X,'abcrobustlogit',M,J,v,s,g,params);
abcboost_save(model,'mymodel.model');
model = abcboost_load('mymodel.model');
res = abcboost_test(testY,testX,model);
