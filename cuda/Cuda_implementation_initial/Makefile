INC="../common/inc"
NVCCFLAGS=-I$(INC) -I
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: cuda_o
cuda_o: N_queens_cuda_unoptimized.cu main_program.cu
		$(NVCC) N_queens_cuda_unoptimized.cu main_program.cu -o cuda_o
clean:
	rm cuda_o
