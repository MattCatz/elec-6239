all: a.out

a.out: hello.c
	mpicc -g hello.c -lm

run: all
	mpirun -n 2 a.out
