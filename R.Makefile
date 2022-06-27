libabcboost_r : src/r_wrapper.cpp src/data.cpp src/tree.cpp src/model.cpp
	PKG_CXXFLAGS='-DOMP -fopenmp' R CMD SHLIB -o libabcboost_r.so src/r_wrapper.cpp src/data.cpp src/tree.cpp src/model.cpp -Isrc/ 
	rm src/*.o
