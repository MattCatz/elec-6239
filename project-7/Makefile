all: a.out

a.out: hello.c
	mpicc -g hello.c -lm

run:
	mpirun -n 3 a.out
