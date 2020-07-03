TARGET=main
OBJECTS=util.o pix2pix.o

CXX=g++
CXXFLAGS=-std=c++11 -Wall -O3 -fopenmp -mmmx -msse -mavx -march=native
# If you use MPI, use the following lines instead of above lines
#CXX=mpic++
#CXXFLAGS=-std=c++11 -Wall -O3 -DUSE_MPI

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

ifndef N
N=64
endif

ARGS="../common/examples/input_$(N).rgb myoutput_$(N).rgb"

ifndef CASE
CASE="run"
endif

run: $(TARGET)
	thorq --add --name $(CASE) ./$(TARGET) ../common/edges2cats_AtoB.bin $(ARGS)
