g++ -O3 -Wall -shared -std=c++11 -fPIC  src/python_wrapper.cpp src/data.cpp src/tree.cpp src/model.cpp -o abcboost.so -Isrc/ $(python3 -m pybind11 --includes) -w
