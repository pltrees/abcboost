cmake_minimum_required(VERSION 3.2)
project(ABCBoost)

if (CUDA)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -fwhole-program")
endif()
set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

file(GLOB SOURCE_FILES src/*.h src/*.cpp)

option(CUDA "cuda" OFF)
option(OMP "omp" ON)

add_definitions(-w)

if (OMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMP -std=c++11 -fopenmp")
endif()

if (CUDA)
  find_package(CUDA 9 REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA -w")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DCUDA -w -Xptxas -O3 -arch=sm_60")
	include_directories(third_parties/)
  CUDA_ADD_LIBRARY(cuda_obj STATIC src/utils.cu src/model_gpu.cu)
endif()

add_executable(abcboost_train src/train_main.cpp src/data.cpp src/tree.cpp src/model.cpp)
add_executable(abcboost_predict src/predict_main.cpp src/data.cpp src/tree.cpp src/model.cpp)

if (CUDA)
  target_link_libraries(abcboost_train ${CUDA_LIBRARIES})
  target_link_libraries(abcboost_train cuda_obj)
  target_link_libraries(abcboost_train ${CUDA_curand_LIBRARY})
  target_link_libraries(abcboost_predict ${CUDA_LIBRARIES})
  target_link_libraries(abcboost_predict cuda_obj)
endif(CUDA)

