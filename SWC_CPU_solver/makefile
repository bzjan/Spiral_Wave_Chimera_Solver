# compiler to use
CC=g++
# add flag "-DDOUBLE" for double or leave it out for single precision
PRECISIONFLAG = -DDOUBLE
CCFLAGS=-Wall $(PRECISIONFLAG) -march=native -std=c++11 -fopenmp

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
name=$(shell basename $(CURDIR)).exe




release: CCFLAGS += -DDEBUG -O3
release: all

# no optimization
no: CCFLAGS += 
no: all

debug: CCFLAGS += -DDEBUG -O0 -g
debug: all

all:$(OBJECTS)
	$(CC) $(OBJECTS) $(CCFLAGS) -lboost_program_options -o $(name)
	rm -rf *o

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.exe *.gch
