mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O3" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O3" abcboost_train.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O0 -g" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O0 -g" abcboost_test.cpp ../data.cpp ../tree.cpp ../model.cpp -I.. 
mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O3" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O3" abcboost_load.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O3" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O3" abcboost_save.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex libsvmread.c

