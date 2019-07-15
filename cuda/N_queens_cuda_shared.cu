#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common/inc/helper_cuda.h"
#include "common/inc/helper_functions.h"

/*
  Static definitions
*/
#define MAX_THREADS 512
#define MIN_BOARDSIZE 2
#define MAX_BOARDSIZE 35
typedef unsigned long long ULL;


/*
  Global variales
*/
int N; // size of the board
int depth; // depth of the cpu solution
int threads_per_block; // number of threads per block
int gpu_index = 0; // index of the device used

/* This function prints how the program is used and what the commands do */
void usage(const char *name){
		printf("Usage: %s [-t threads_per_block] [-d depth] N \n\n", name);
		printf("Options:\n");
		printf("\t-t Used to set the count of threads for each block. [1..512]\n");
		printf("\t-d Indicates the depth to which the solutions shall be precalculated [Must be less than N].\n");
		printf("\t   Size of the board [N] \n");
		exit(0);
}

/* This function checks the validity of the arguments that are passed in and exits the program if they are not with a message */
void check_arguments(){
	if (threads_per_block < 1|| threads_per_block > MAX_THREADS){
        printf("Threads per Block must be between [%d and %d]\n", 1, 512);
        exit(0);
  }
	if (N < MIN_BOARDSIZE || N > MAX_BOARDSIZE){
        printf("Queen count must be between [%d and %d]\n", MIN_BOARDSIZE, MAX_BOARDSIZE );
        exit(0);
  }

	if (N < 1|| depth > (N-1)){
        printf("Depth must be between [%d and %d)\n", 1, N);
        exit(0);
  }
}
/* This function checks the flag passed in and sets the input to the correct variables */
void parse_arguments(int * arg_index, int argc, char *argv[]){
	if(strcmp(argv[*arg_index], "-t") == 0){
		(*arg_index)++;
		threads_per_block = atoi(argv[*arg_index]);
		return;
	}

	if(strcmp(argv[*arg_index], "-d") == 0){
		(*arg_index)++;
		depth = atoi(argv[*arg_index]);
		return;
	}
	if(*arg_index == argc - 1){
    N = atoi(argv[*arg_index]);
  }
	else{
    usage(argv[0]);
  }
}

void parse_command_line_arguments(int argc, char *argv[]){
	if(argc < 2){
    // if there are not enough arguments
    usage(argv[0]);
  }

	for(int i = 1; i < argc; i++){
    parse_arguments(&i, argc, argv);
    // reading in the command line arguments
  }
  // checking to see if the arguments passed in are valid
	check_arguments();
}

/*
	Function which calculates the partial solution of the
	n_queens problem for a given depth d and returns the valid
	thread count in order to complete the solutions using cuda

*/
int init_n_queens(int board_size, int depth, int *data_cpu, unsigned int data_length){
	int thread_count = 0;

  int a_queen_bit_res[MAX_BOARDSIZE]; // results
  int a_queen_bit_col[MAX_BOARDSIZE]; // marks columns which already have queens
  int a_queen_bit_pos_diag[MAX_BOARDSIZE]; // marks "positive diagonals" which already have queens
  int a_queen_bit_neg_diag[MAX_BOARDSIZE]; // marks "negative diagonals" which already have queens

  int a_stack[MAX_BOARDSIZE + 2]; // we use a stack instead of recursion
	int *pn_stack;

	int num_rows = 0;
	unsigned int lsb; // least significant bit
	unsigned int bitfield; // bits which are set mark possible positions for a queen

	int i;
  int odd = board_size & 1; // 0 if board_size even, 1 if odd
  int board_minus = depth;
  int mask = (1 << board_size) - 1; // mask consists of N 1's

  // Initialize stack
  a_stack[0] = -1; // setting the end position of the array

  /*
	 	(board_size & 1) is true if board_size is odd
  	We need to loop through 2x if board_size is odd
	*/
  for (i = 0; i < (1 + odd); i++){
      bitfield = 0;
      if (i == 0){
          /* Handle half of the board, except the middle
             column. So if the board is 5 x 5, the first
             row will be: 00011, since we're not worrying
             about placing a queen in the center column (yet).
          */
          int half = board_size >> 1; // divide by two
          /* fill in rightmost 1's in bitfield for half of board_size
             If board_size is 7, half of that is 3 (we're discarding the remainder)
             and bitfield will be set to 111 in binary. */
          bitfield = (1 << half) - 1;
          pn_stack = a_stack + 1; // stack pointer, first position

          a_queen_bit_res[0] = 0;
          a_queen_bit_col[0] = a_queen_bit_pos_diag[0] = a_queen_bit_neg_diag[0] = 0;
      }else{
          /* Handle the middle column (of a odd-sized board).
             Set middle column bit to 1, then set
             half of next row.
             So we're processing first row (one element) & half of next.
             So if the board is 5 x 5, the first row will be: 00100, and
             the next row will be 00011.
          */
          bitfield = 1 << (board_size >> 1);
          num_rows = 1; // prob already 0

          // The first row just has one queen (in the middle column)
          a_queen_bit_res[0] = bitfield;
          a_queen_bit_col[0] = a_queen_bit_pos_diag[0] = a_queen_bit_neg_diag[0] = 0;
          a_queen_bit_col[1] = bitfield;

          /* Now do the next row.  Only set bits in half of it, because we'll
             flip the results over the "Y-axis".  */
          a_queen_bit_neg_diag[1] = (bitfield >> 1);
          a_queen_bit_pos_diag[1] = (bitfield << 1);
          pn_stack = a_stack + 1; /* stack pointer */
          *pn_stack++ = 0; // we're done with this row -- only 1 element & we've done it
					bitfield = (bitfield - 1) >> 1; // bitfield -1 is all 1's to the left of the single 1
      }

      /* this is the critical loop */
      for (;;)
      {
          lsb = -((signed)bitfield) & bitfield; // this assumes a 2's complement architecture
          if (0 == bitfield)
          {
              bitfield = *--pn_stack; // get prev. bitfield from stack
              if (pn_stack == a_stack) { // if we hit the end of the stack
                  break; // break out of the critical loop
              }
              --num_rows; // reduces the number of rows for queens to be placd
              continue; // ensuring loop continues
          }

          bitfield &= ~lsb; // toggle off this bit so we don't try it again

          a_queen_bit_res[num_rows] = lsb; // save the result
          if (num_rows < board_minus){ // if we still have more rows in the board to process
              int n = num_rows++;
              a_queen_bit_col[num_rows] = a_queen_bit_col[n] | lsb;
              a_queen_bit_neg_diag[num_rows] = (a_queen_bit_neg_diag[n] | lsb) >> 1;
              a_queen_bit_pos_diag[num_rows] = (a_queen_bit_pos_diag[n] | lsb) << 1;
              *pn_stack++ = bitfield;
              /*
							   We can't consider positions for the queen which are in the same
                 column, same positive diagonal, or same negative diagonal as another
                 queen already on the board.
							*/
              bitfield = mask & ~(a_queen_bit_col[num_rows] | a_queen_bit_neg_diag[num_rows] | a_queen_bit_pos_diag[num_rows]);
              continue;
          }else{
            // we reached a solution
						for(int i = 0; i < depth; i++){
							unsigned int address = thread_count * depth + i;
							if(address > data_length){
								printf("Internal Error !");
								exit(1);
							}

							data_cpu[address] = a_queen_bit_res[i] ^ (a_queen_bit_res[i] & (a_queen_bit_res[i] - 1));
						}

						++thread_count;

            bitfield = *--pn_stack;
            --num_rows;
            continue;
         }
      }
    }
	return thread_count;
}
/*
	get the allocation size for the thread count
	and for the given depth.
*/
unsigned int get_allocation_size(int thread_count, int depth){
	return (thread_count + 1) * depth;
}

//  function which initialises the data arrays
int* init_data(int *thread_count, int size, int depth){
	// guess allocation size
	unsigned int allocation_size = size;
	for(int i = 0; i < (depth - 1); i++){
		allocation_size *= (size - i);
	}

	// allocates memory for the data array based on the allocation size
	int *data = (int*)malloc(allocation_size * sizeof(int));
	if(data == 0){
		printf("Error allocating memory\n");
		exit(-1);
	}
	*thread_count = init_n_queens(size, depth, data, allocation_size);

	// gets the allocation size based on the thread counted returned by init_n_queens
	allocation_size = get_allocation_size(*thread_count, depth);

	// creates array based on allocation size (computed using thread_count and depth)
	int *result = (int*)malloc(allocation_size * sizeof(int));
	for(unsigned int i = 0; i < allocation_size; i++){
		result[i] = data[i];
	}
	// frees the memory
	free(data);
	// returns the partial solutions array
	return result;
}
/*
	Kernel function which computes the full solution for
	partial solution
*/
__global__ void n_queens_complete_solution(int board_size, int thread_count, int depth, int *data_gpu){
	extern __shared__ int shared_data[];

	int thread_x = threadIdx.x;

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread_id >= thread_count || thread_id < 0){
		return;
	}

	int remaining = board_size - depth; // remainder of board to be solved
	int i_queen_bit_col = remaining * 4 * thread_x + 0 * remaining;
	int i_queen_bit_pos_diag = remaining * 4 * thread_x + 1 * remaining;
	int i_queen_bit_neg_diag = remaining * 4 * thread_x + 2 * remaining;
	int i_stack = remaining * 4 * thread_x + 3 * remaining;

	// variable to keep track of the total number of solutions
	ULL total_solutions = 0;

	int n_stack = i_stack;
  int num_rows = 0;

  unsigned int lsb; // least significant bit
  unsigned int bitfield; // bits which are set mark possible positions for a queen
  int board_minus = board_size - 1; // board size - 1
  int mask = (1 << board_size) - 1; // if board size is N, mask consists of N 1's

  shared_data[n_stack] = -1; // initialize stack -1 represents the end
  bitfield = 0; // setting the bitfield to 0

  /* Handle half of the board, except the middle
     column. So if the board is 5 x 5, the first
     row will be: 00011, since we're not worrying
     about placing a queen in the center column (yet).
  */
  int half = board_size >> 1; // divide by two
  /* fill in rightmost 1's in bitfield for half of board_size
     If board_size is 7, half of that is 3 (we're discarding the remainder)
     and bitfield will be set to 111 in binary. */
  bitfield = (1 << half) - 1;
  n_stack += 1; // pointer to current location in stack

  shared_data[i_queen_bit_col] = shared_data[i_queen_bit_pos_diag] = shared_data[i_queen_bit_neg_diag] = 0;

	for(int d = 0; d < depth; d++){
		lsb = data_gpu[thread_id * depth + d];
		bitfield &= ~lsb;

    shared_data[i_queen_bit_col] = shared_data[i_queen_bit_col] | lsb;
		shared_data[i_queen_bit_pos_diag] = (shared_data[i_queen_bit_pos_diag] | lsb) << 1;
    shared_data[i_queen_bit_neg_diag] = (shared_data[i_queen_bit_neg_diag] | lsb) >> 1;
    shared_data[n_stack] = bitfield;
    /* We can't consider positions for the queen which are in the same
       column, same positive diagonal, or same negative diagonal as another
       queen already on the board. */
    bitfield = mask & ~(shared_data[i_queen_bit_col] | shared_data[i_queen_bit_pos_diag] | shared_data[i_queen_bit_neg_diag]);
	}

  /* this is the critical loop */
  for (;;){
    lsb = -((signed)bitfield) & bitfield; //this assumes a 2's complement architecture
    if (0 == bitfield)
    {
        bitfield = shared_data[--n_stack]; // get previous bitfield from stack
        if (n_stack == i_stack) { // at end of stack
            break ;
        }
        --num_rows;
        continue;
    }
    bitfield &= ~lsb; // toggle off this bit so we don't try it again

    //aQueenBitRes[numrows] = lsb; // save the result
    if (num_rows < board_minus - depth){
			int n = num_rows++;
	    shared_data[i_queen_bit_col + num_rows] = shared_data[i_queen_bit_col + n] | lsb;
			shared_data[i_queen_bit_pos_diag + num_rows] = (shared_data[i_queen_bit_pos_diag + n] | lsb) << 1;
			shared_data[i_queen_bit_neg_diag + num_rows] = (shared_data[i_queen_bit_neg_diag + n] | lsb) >> 1;
	    shared_data[n_stack++] = bitfield;
	    /* We can't consider positions for the queen which are in the same
	       column, same positive diagonal, or same negative diagonal as another
	       queen already on the board. */
	    bitfield = mask & ~(shared_data[i_queen_bit_col + num_rows] | shared_data[i_queen_bit_neg_diag + num_rows] | shared_data[i_queen_bit_pos_diag + num_rows]);
	    continue;
    }else {
      // reached a solution
      ++total_solutions;
      bitfield = shared_data[--n_stack];
      --num_rows;
      continue;
    }
  }
  // multiply solutions by two, to count mirror images
	data_gpu[thread_id * depth] =  total_solutions * 2;
}
/*
	Calculates the amount of shared memory needed
*/
int shared_memory_needed(int board_size, int threads_per_block){
	return board_size * 4 * threads_per_block * sizeof(int) // memory for 4 int arrays
		+ 2 * threads_per_block * sizeof(int); // stack array is 2 columns bigger
}

ULL calculate_full_solutions(int gpuIndex, int board_size, int depth, int threads_per_block){
	int thread_count = 0;
	int *data_cpu = init_data(&thread_count, board_size, depth);
	int block_size = (thread_count/threads_per_block)+1;
	ULL total_solutions = 0;
	// create events for time measurement
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaSetDevice(gpuIndex));
	int memory_size = get_allocation_size(thread_count, depth) * sizeof(int);
	int *data_gpu;
	checkCudaErrors(cudaMalloc((void**)&data_gpu, memory_size));
	checkCudaErrors(cudaMemcpy(data_gpu, data_cpu, memory_size, cudaMemcpyHostToDevice));
	int shared_memory_size = shared_memory_needed(board_size - depth, threads_per_block);

	checkCudaErrors(cudaEventRecord(start));
	// complete partial solution
	n_queens_complete_solution<<<block_size, threads_per_block, shared_memory_size>>>(board_size, thread_count, depth, data_gpu);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(stop));
	// checkCudaErrors(cudaThreadSynchronize());
	checkCudaErrors(cudaMemcpy(data_cpu, data_gpu, memory_size, cudaMemcpyDeviceToHost));

	// time taken for kernel to run
	float time;
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	printf("\nKernel completion time: %f seconds\n\n", time/1000.0);
	// checkCudaErrors(cudaThreadExit());
	for(int thread = 0; thread < thread_count; thread++){
		// printf("Total solutions: %llu\n", total_solutions);
		total_solutions += data_cpu[thread * depth];
	}

	return total_solutions;
}

/*
  Main function
*/
int main(int argc, char** argv)
{
	clock_t start, stop;
  parse_command_line_arguments(argc, argv);
	start = clock();
	ULL total_solutions = calculate_full_solutions(gpu_index, N, depth, threads_per_block);
	stop = clock();
  double time = ((double)(stop-start)/CLOCKS_PER_SEC);
  printf("Total number of solutions: %llu\n", total_solutions);
  printf("\nRunning time for placing %d queens on a %d x %d board %lf seconds\n", N, N, N, time);
  return 0;
}
