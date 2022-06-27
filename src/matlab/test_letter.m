tr = load('../../data/letter10k.train');
te = load('../../data/letter10k.test');
Y = tr(:,1);
X = tr(:,2:end);
testY = te(:,1);
testX = te(:,2:end);

params = struct;
params.use_omp = 0;
params.n_threads = 1;
model = abcboost_train(Y,X,'abcrobustlogit',10,20,0.1,1,0,params);
%abcboost_save(model,'mymodel.model');
%model = abcboost_load('mymodel.model');
res = abcboost_test(testY,testX,model);
