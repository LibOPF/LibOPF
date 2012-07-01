LIB=./lib
INCLUDE=./include
SRC=./src
OBJ=./obj

CC=gcc

FLAGS= -march=native -O3 -Wall -fPIC -fopenmp -D NTHREADS=4 -lgomp

INCFLAGS = -I$(INCLUDE)

all: libopf

libopf: libopf-build

libopf-build: \
aux

	ar rcs $(LIB)/libopf.a $(OBJ)/*.o

aux: $(SRC)/common.c $(SRC)/set.c $(SRC)/realheap.c $(SRC)/linearalloc.c  $(SRC)/metrics.c  $(SRC)/measures.c $(SRC)/graph.c $(SRC)/knn.c $(SRC)/supervised.c $(SRC)/unsupervised.c
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/common.c       -o $(OBJ)/common.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/set.c          -o $(OBJ)/set.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/realheap.c     -o $(OBJ)/realheap.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/linearalloc.c  -o $(OBJ)/linearalloc.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/metrics.c      -o $(OBJ)/metrics.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/measures.c     -o $(OBJ)/measures.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/graph.c        -o $(OBJ)/graph.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/knn.c          -o $(OBJ)/knn.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/supervised.c   -o $(OBJ)/supervised.o
	$(CC) $(FLAGS) $(INCFLAGS) -c $(SRC)/unsupervised.c -o $(OBJ)/unsupervised.o

## Cleaning-up

clean:
	rm -f $(LIB)/lib*.so; rm -f $(OBJ)/*.o

cython:
	cython libopf_py.pyx

bindings:
	$(CC) -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing $(INCFLAGS) -I/usr/include/python2.7 -fopenmp -o $(LIB)/libopf_py.so libopf_py.c $(LIB)/libopf.a
