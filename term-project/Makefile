.PHONY = all

all: term_mpi

term_mpi: main_mpi.c term.h
	mpicc -Wall -g $^  -lm
