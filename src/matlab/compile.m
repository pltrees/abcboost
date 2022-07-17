mex CXXFLAGS="$CXXFLAGS -O3" LDFLAGS="$LDFLAGS -O3" abcboost_train.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex CXXFLAGS="$CXXFLAGS -O3" LDFLAGS="$LDFLAGS -O3" abcboost_test.cpp ../data.cpp ../tree.cpp ../model.cpp -I.. 
mex CXXFLAGS="$CXXFLAGS -O3" LDFLAGS="$LDFLAGS -O3" abcboost_load.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex CXXFLAGS="$CXXFLAGS -O3" LDFLAGS="$LDFLAGS -O3" abcboost_save.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex CXXFLAGS="$CXXFLAGS -O3" LDFLAGS="$LDFLAGS -O3" abcboost_predict.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex libsvmread.c


