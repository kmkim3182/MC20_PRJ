TARGET=main
CUDA_INC_DIR=/usr/local/cuda/include
CUDA_LIB_DIR=/usr/local/cuda/lib64
MPI_DEFINE=USE_MPI

SRC_DIR=src/
OBJ_DIR=obj/

CXXSOURCE=util.cpp control.cpp pix2pix.cpp main.cpp
CUDASOURCE=pix2pix_v2.cu

CXXOBJ=util.o control.o pix2pix.o main.o
CUDAOBJ=pix2pix_v2_cu.o
CUDALINKOBJ=pix2pix_v2.dlink.o

VPATH=$(SRC_DIR):$(OBJ_DIR)
OBJS=$(addprefix $(OBJ_DIR), $(CXXOBJ) $(CUDAOBJ) $(CUDALINKOBJ))

NVCC=nvcc

CXX=mpic++
CXXFLAGS=-std=c++11 -Wall -O3 -mmmx -msse -mavx -march=native -I$(CUDA_INC_DIR) -D$(MPI_DEFINE)
LDFLAGS=-lm -L$(CUDA_LIB_DIR) -L$(MPI_LIB) -lcudart -lcudadevrt -D$(MPI_DEFINE)
CUDAFLAGS=-std=c++11 -O3 -w -use_fast_math -rdc=true -gencode arch=compute_60,code=sm_60 -I$(CUDA_INC_DIR) -I$(MPI_INCLUDE) -D$(MPI_DEFINE)

#CXX=g++
#CXXFLAGS=-std=c++11 -Wall -O3 -mmmx -msse -mavx -march=native -I$(CUDA_INC_DIR)
#LDFLAGS=-lm -L$(CUDA_LIB_DIR) -L$(MPI_LIB) -lcudart -lcudadevrt
#CUDAFLAGS=-std=c++11 -O3 -w -use_fast_math -rdc=true -gencode arch=compute_60,code=sm_60 -I$(CUDA_INC_DIR)

all: $(TARGET)

$(TARGET): $(CXXOBJ) $(CUDAOBJ) $(CUDALINKOBJ)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) $(CXXFLAGS)

$(CXXOBJ):
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)$(@:.o=.cpp) -o $(OBJ_DIR)$@

$(CUDAOBJ):
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CUDAFLAGS) -c $(SRC_DIR)$(@:_cu.o=.cu) -o $(OBJ_DIR)$@

$(CUDALINKOBJ): 
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CUDAFLAGS) -dlink $(OBJ_DIR)$(@:.dlink.o=_cu.o) -o $(OBJ_DIR)$@ $(LDFLAGS)

clean:
	rm -rf $(TARGET)
	rm -rf $(OBJ_DIR)*.*

util.o : util.h util.cpp
control.o : control.h control.cpp
pix2pix_v2_cu.o : control.h control.cuh operator.cuh pix2pix_v2.cu
pix2pix_v2.dlink.o : pix2pix_v2_cu.o
pix2pix.o : pix2pix.h pix2pix.cpp
main.o : util.h pix2pix.h main.cpp

ifndef N
N=2048
endif

ifeq ($(MAKECMDGOALS), validation)
ARGS="../common/examples/input_$(N).rgb ./validation/myoutput_$(N).rgb"
else
ARGS="../common/examples/input_$(N).rgb myoutput_$(N).rgb"
endif

ifndef CASE
CASE="run"
endif

ifndef SLOTS
SLOTS=4
endif

ifndef NODES
NODES=4
endif

.PHONY: validation

validation: $(TARGET)
	@mkdir -p validation
	thorq --add --mode mpi --nodes $(NODES) --slots $(SLOTS) --name $(CASE) --device gpu/1080 ./$(TARGET) ../common/edges2cats_AtoB.bin $(ARGS)

run: $(TARGET)
	thorq --add --mode mpi --nodes $(NODES) --slots $(SLOTS) --name $(CASE) --device gpu/1080 ./$(TARGET) ../common/edges2cats_AtoB.bin $(ARGS)
