.PHONY = all

EXECUTABLE = term_cuda

all: term_mpi

term_mpi: main_mpi.c term.h
	mpicc -Wall -std=c99 -g $^ -o $@ -lm

include cuda.makefile