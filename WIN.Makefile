CC=g++
FLAGS=-DOMP -fopenmp -O3 -march=native -flto -fwhole-program

all : train

train : src/train_main.cpp src/data.cpp src/tree.cpp src/model.cpp src/*.h
	$(CC) $(FLAGS) -o train src/train_main.cpp src/data.cpp src/tree.cpp src/model.cpp -Isrc/ 
