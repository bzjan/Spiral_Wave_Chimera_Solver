# compiler to use
CC=g++
NCC=nvcc

# linker flag at compile time
LDFLAGS=-L/usr/local/cuda/lib64
CCFLAGS = -Wall -std=c++11
NCCFLAGS = -std=c++11
PPFLAGS= -DDEBUG
# add -DDOUBLE for double precision or nothing for float
PRECISIONFLAG = -DDOUBLE
SOURCES = $(wildcard *.cpp)
GPUSOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cpp=.o) $(GPUSOURCES:.cu=_gpu.o)
name=$(shell basename "$(CURDIR)").exe
HOST=$(shell hostname)

# set compute capability
# custom
cc=50
# general
ifeq ($(cc),)
	SMS = 20 30 35 37 50 52 60
	$(foreach sm,$(SMS),$(eval GPUARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))
else
	GPUARCH = -gencode arch=compute_$(cc),code=sm_$(cc)
endif



release: CCFLAGS += -O3 $(PRECISIONFLAG)
release: NCCFLAGS += -O3 $(PRECISIONFLAG)
release: all

# not applicable for zbke2k
fast: CCFLAGS += -O3
fast: NCCFLAGS += -O3 -use_fast_math
fast: all

# no optimization
no: CCFLAGS += $(PRECISIONFLAG)
no: NCCFLAGS += $(PRECISIONFLAG) -prec-div=true -prec-sqrt=true
no: all

debug: CCFLAGS += -g -O0 $(PRECISIONFLAG)
debug: NCCFLAGS += -g -G -lineinfo $(PRECISIONFLAG)
debug: all

all:$(OBJECTS)
	$(NCC) $(GPUARCH) $(NCCFLAGS) -dlink $(OBJECTS) -o gpu_final.o
	$(CC) gpu_final.o $(OBJECTS) $(CCFLAGS) -I/usr/local/cuda/include -lcudart -lboost_program_options -o $(name) $(LDFLAGS)
	rm -rf *o

%_gpu.o: %.cu
	$(NCC) -x cu $(PPFLAGS) $(NCCFLAGS) $(GPUARCH) -dc $< -o $@

%.o: %.cpp
	$(CC) $(PPFLAGS) $(CCFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.exe *.gch
