INC="../common/inc"
NVCCFLAGS=-I$(INC)
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: serial

serial: serial_n_queens.c
		$(CC) $(CCFLAGS) serial_n_queens.c -o serial

clean:
	rm serial
