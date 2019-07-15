#include "N_queens_cuda_unoptimized.h"
#include <cuda_runtime.h>
#include "common/inc/helper_cuda.h"
#include "common/inc/helper_functions.h"

unsigned int guess_allocation_size(int size, int depth){
  int result = size;

  for(int i = 0; i < depth - 1; i++){
    result *= (size - 1);
  }

  return result;
}

 unsigned int get_allocation_size(int thread_count, int depth){
   return (thread_count + 1)*depth;
 }

 int init_data_using_nqueens(int board_size, int depth, int *data, unsigned int data_length){

   int thread_count = 0;

   // results
   int a_queen_bit_res[MAX_BOARDSIZE];
   // marks the columns which already have queens
   int a_queen_bit_col[MAX_BOARDSIZE];
   // marks "positice diagonals"  which already have queens
   int a_queen_bit_pos_diag[MAX_BOARDSIZE];
   // marks "negative diagonals" which already have queens
   int a_queen_bit_neg_diag[MAX_BOARDSIZE];
   // using a stack to keep track of queen placements instead of a recursive method
   int a_stack[MAX_BOARDSIZE + 2];


   register int *pn_stack;
   // could use a stack
   register int num_rows = 0;
   // least significant bit
   register unsigned int lsb;
   // bits which are set mark possible for a queen
   register unsigned int bitfield;

   int i;
   // 0 if board size is even, 1 is board size is odd
   int odd = board_size & 1;
   // board size - 1
   int board_minus = depth;
   // board size is N mask consists of N 1's
   int mask = (1 << board_size) -1;


   // initializing a stack

   a_stack[0] = -1;  // this values signifies the end of the stack

   // if the board size is odd
   for(i = 0; i < (1 + odd); i++){
     bitfield = 0;
     // printf("%d\n", i);
     if(i == 0){
       // handle half the board except the middle column
       int half  = board_size >> 1; // divide by 2
       // filling in the board in the right mose location
       bitfield = (1 << half) - 1;
       // stack pointer
       pn_stack = a_stack + 1;

       a_queen_bit_res[0] = 0;
       a_queen_bit_col[0] = a_queen_bit_pos_diag[0] = a_queen_bit_neg_diag[0] = 0;
     }else{
       // handles the middle column of the board
       bitfield = 1 << (board_size >> 1);
       num_rows = 1;

      // the first row just has one queen
      a_queen_bit_res[0] = bitfield;
      a_queen_bit_col[0] = a_queen_bit_pos_diag[0] = a_queen_bit_neg_diag[0] = 0;
      a_queen_bit_col[1] = bitfield;

      // now we do the next row, the results will flip over
      a_queen_bit_neg_diag[1] = (bitfield >> 1);
      a_queen_bit_pos_diag[1] = (bitfield << 1);
      // stack pointer
      pn_stack = a_stack + 1;
      *pn_stack++ = 0;
      bitfield = (bitfield - 1) >> 1;
     }
     // this is the critical loop
     for(;;){
       // we want to get the first least significant bit

       lsb = -((signed)bitfield) & bitfield; // assumes a twos complement architecture
       if(0 == bitfield){
         bitfield = *--pn_stack; // getting the previous bit field from the stack
         if(pn_stack == a_stack){ // if we reach the end of the stack
           break;
         }
         --num_rows;
         continue;
       }
       // putting this bit off so we don't use it again
       bitfield &= ~lsb;
       a_queen_bit_res[num_rows] = lsb; // saving the result
       if(num_rows < board_minus){
         // if we still have more rows to process
         int n = num_rows++;
         a_queen_bit_col[num_rows] = a_queen_bit_col[n] | lsb;
         a_queen_bit_neg_diag[num_rows] = (a_queen_bit_neg_diag[n] | lsb) >> 1;
         a_queen_bit_pos_diag[num_rows] = (a_queen_bit_pos_diag[n] | lsb) << 1;

         *pn_stack++ = bitfield;
         // we don't want to consider queens which are attacking each other
         bitfield = mask & ~(a_queen_bit_col[num_rows] | a_queen_bit_neg_diag[num_rows] | a_queen_bit_pos_diag[num_rows]);
         continue;
       }else{
         // we have no more rows to process, we found found a solution

         // print_table(board_size, a_queen_bit_res, solution_count + 1);
        for(int i = 0; i < depth; i++){
          unsigned int address = thread_count * depth + 1;
          if(address > data_length){
            printf("Internal error\n");
            exit(-1);
          }
          data[address] = a_queen_bit_res[i] ^ (a_queen_bit_res[i] & (a_queen_bit_res[i] - 1));
        }
        thread_count++;
        bitfield = *--pn_stack;
        num_rows--;
        continue;
       }
     }
   }
   return thread_count;
}

int* malloc_int(int size){
  int *result = (int *)malloc(size * sizeof(int));
  if(result == 0){
    printf("Error allocating memory\n");
    exit(-1);
  }
  return result;
}
int * init_data(int *thread_count, int size, int depth){

  unsigned int allocation_size = guess_allocation_size(size, depth);

  int *data = malloc_int(allocation_size);
  *thread_count = init_data_using_nqueens(size, depth, data, allocation_size);
  allocation_size = get_allocation_size(*thread_count, depth);
  // printf("This is the size of the data: %u\n", allocation_size);
  int *result = malloc_int(allocation_size);
  for(unsigned int i = 0; i < allocation_size; i++){
    result[i] = data[i];
  }
  free(data);
  return result;
}
/*
  Kernel funciton which solves the nqueens problem with backtracking. It places a queen in a given row
  and marks the diagonal and columns aligned with it as blocked. It then tries to place a queen in the
  valid blocks in the next row and if it can't place the queen there then it will backtrack and move the
  previously placed queen to its next valid position.
*/
__global__ void n_queens_from_depth(int board_size, int thread_count, int depth, int *data){

   int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

   if(thread_id >= thread_count || thread_id < 0){
     return;
   }
   ULL solution_count = 0;

   // mark columns and diagonals that are set
   int q_bit_col[MAX_BOARDSIZE];
   int q_bit_pos_diag[MAX_BOARDSIZE];
   int q_bit_neg_diag[MAX_BOARDSIZE];

   // definition of stack
   int a_stack[MAX_BOARDSIZE + 2];
   register int n_stack;

   register int num_rows = 0;
   // least significant bit
   register unsigned int lsb;

   // bits which are set mark free positions
   register unsigned int bitfield = 0;
   register int board_minus = board_size - 1;
   // if board size is N mask consists of N 1's
   register int mask = (1 << board_size) - 1;
   // initialize the stack
   a_stack[0] = -1; // this represents the end of the stack
   int half = board_size >> 1; // divides by 2
   // filling in the rightmost ones
   bitfield = (1 << half) - 1;

   n_stack = 1; // stack pointer
   q_bit_col[0] = q_bit_pos_diag[0] = q_bit_neg_diag[0] = 0;
   // initialize the field

   for(;num_rows < depth;){
     lsb = data[thread_id * depth + num_rows];
     bitfield &= ~lsb;

     int n = num_rows++;

     // mark occupied places in the next row
     q_bit_col[num_rows] = q_bit_col[n] | lsb;
     q_bit_neg_diag[num_rows] = (q_bit_neg_diag[n] | lsb) >> 1;
     q_bit_pos_diag[num_rows] = (q_bit_pos_diag[n] | lsb) << 1;

     a_stack[n_stack++] = bitfield;
     // not considering queens that are already placed
     bitfield = mask & ~(q_bit_col[num_rows] | q_bit_neg_diag[num_rows] | q_bit_pos_diag[num_rows]);
   }

   // this is the critical loop
   for(;;){
     // get the first lsb
    lsb = -((signed)bitfield) & bitfield;
    if(0 == bitfield){
      bitfield = a_stack[--n_stack];
      if(n_stack == depth){
        break;
      }
      --num_rows;
      continue;
    }
    bitfield &= ~lsb; // switch this bit off

    // if there are still more rows
    if(num_rows < board_minus){
      int n = num_rows++;
      // marking the occupied places in the next row
      q_bit_col[num_rows] = q_bit_col[n] | lsb;
      q_bit_neg_diag[num_rows] = (q_bit_neg_diag[n] | lsb) >> 1;
      q_bit_pos_diag[num_rows] = (q_bit_pos_diag[n] | lsb) << 1;

      a_stack[n_stack++] = bitfield;
      // we cannot consider positions of queens which are already placed
      bitfield = mask & ~(q_bit_col[num_rows] | q_bit_neg_diag[num_rows] | q_bit_pos_diag[num_rows]);
      continue;
    }else{
      // no more rows we have reached a solution
      ++solution_count;
      bitfield = a_stack[--n_stack];
      --num_rows;
      continue;
    }
   }
   // multiply solutions by 2 to mirror images
   data[thread_id*depth] = solution_count * 2;
}
ULL calculate_solutions(int gpu_index, int board_size, int depth, int threads_per_block, bool verbose){
  int thread_count = 0;
  int *data = init_data(&thread_count, board_size, depth);
  int block_size = (thread_count/threads_per_block) + 1;

   ULL solution_count = 0;

   // creating events for time measurements
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   int mem_size = get_allocation_size(thread_count, depth) * sizeof(int);
   int *data_gpu;
   checkCudaErrors(cudaMalloc((void**)&data_gpu, mem_size));
   checkCudaErrors(cudaMemcpy(data_gpu, data, mem_size, cudaMemcpyHostToDevice));
   cudaEventRecord(start);
   n_queens_from_depth<<<block_size, threads_per_block>>>(board_size, threads_per_block, depth, data_gpu);
   checkCudaErrors(cudaDeviceSynchronize());
   cudaEventRecord(stop);
   checkCudaErrors(cudaMemcpy(data, data_gpu, mem_size, cudaMemcpyDeviceToHost));
   float time;
   cudaEventElapsedTime(&time, start, stop);
   printf("\n Completion time: %f ms\n\n", time);
   for(int thread = 0; thread < thread_count; thread++){
     // printf("%llu\n", solution_count);
     solution_count += data[thread * depth];
   }
   return solution_count;
}
