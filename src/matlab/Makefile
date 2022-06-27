#Set the mex path if MATLAB bin directory is not exported in the PATH environment variable.
#For example: a linux matlab mex path MIGHT be /usr/local/MATLAB/R2019a/bin/mex
MEX = mex

all: abcboost_train abcboost_test abcboost_save abcboost_load

dep : ../data.cpp ../tree.cpp ../model.cpp ../data.h ../tree.h ../model.h

abcboost_train: abcboost_train.cpp dep
	$(MEX) abcboost_train.cpp ../data.cpp ../tree.cpp ../model.cpp -I..

abcboost_test: abcboost_test.cpp dep
	$(MEX) abcboost_test.cpp ../data.cpp ../tree.cpp ../model.cpp -I..

abcboost_save: abcboost_save.cpp dep
	$(MEX) abcboost_save.cpp ../data.cpp ../tree.cpp ../model.cpp -I..

abcboost_load: abcboost_load.cpp dep
	$(MEX) abcboost_load.cpp ../data.cpp ../tree.cpp ../model.cpp -I..

clean:
	rm -f *~ *.o *.mex* *.obj
