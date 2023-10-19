cmake_minimum_required(VERSION 3.12)
project(ABCBoost)

if (CUDA)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -fwhole-program")
endif()
set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

file(GLOB SOURCE_FILES src/*.h src/*.cpp)

option(CUDA "cuda" OFF)
option(OMP "omp" OFF)
option(NATIVE "native" OFF)

if (NATIVE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()


add_definitions(-w)

if (OMP)
  find_package(OpenMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMP -std=c++11")
endif()

if (CUDA)
  find_package(CUDA 9 REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA -w")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DCUDA -w -Xptxas -O3 -arch=sm_60")
  CUDA_ADD_LIBRARY(cuda_obj STATIC src/utils.cu src/model_gpu.cu)
endif()

add_executable(abcboost_train src/train_main.cpp src/data.cpp src/tree.cpp src/model.cpp)
add_executable(abcboost_predict src/predict_main.cpp src/data.cpp src/tree.cpp src/model.cpp)
add_executable(abcboost_clean src/clean_main.cpp src/data.cpp)

if (OMP AND OpenMP_CXX_FOUND)
  include_directories(OpenMP_CXX_INCLUDE_DIRS)
  target_link_libraries(abcboost_train PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(abcboost_predict PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(abcboost_clean PUBLIC OpenMP::OpenMP_CXX)
endif()

if (CUDA)
  target_link_libraries(abcboost_train PUBLIC ${CUDA_LIBRARIES})
  target_link_libraries(abcboost_train PUBLIC cuda_obj)
  target_link_libraries(abcboost_train PUBLIC ${CUDA_curand_LIBRARY})
  target_link_libraries(abcboost_predict PUBLIC ${CUDA_LIBRARIES})
  target_link_libraries(abcboost_predict PUBLIC cuda_obj)
endif(CUDA)

