NUMPY_PATH=$(HOME)/.local/lib/python3.6/site-packages/numpy

all:
	cython --cplus parallelSort.pyx  
	g++  -g -march=native -Ofast -fpic   -c parallelSort.cpp -o parallelSort.o -fopenmp `python3-config --includes` -I$(NUMPY_PATH)/core/include/
	g++  -g -march=native -Ofast -shared -o parallelSort.so parallelSort.o `python3-config --libs` -lgomp 

clean:
	rm -f parallelSort.cpp *.o *.so

