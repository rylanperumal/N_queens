INC="../common/inc"
NVCCFLAGS=-I$(INC)
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: serial_rec serial_it

serial_rec: serial_n_queens.c
		$(CC) $(CCFLAGS) serial_n_queens.c -o serial_rec
serial_it: N_queens_iterative.c
		$(CC) $(CCFLAGS) N_queens_iterative.c -o serial_it
clean:
	rm serial_rec serial_it
