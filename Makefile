INC="../common/inc"
NVCCFLAGS=-I$(INC)
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: serial
		$(CC) $(CCFLAGS) N_queens_serial.c -o serial
clean:
	rm serial
