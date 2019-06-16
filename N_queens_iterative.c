#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "stack.h"


/*
  Function definitions
*/
void usage(char prog_name[]);
void initialize_board(int ** board, int n);
void print_solution(int **board, int n);
bool check_is_safe(int **board, int row, int col, int n);
void N_queens_it(int **board, int n, struct Stack *i_stack, struct Stack *j_stack);
int pop(struct Stack *stack);
void push(struct Stack *stack, int item);
bool is_empty(struct Stack *stack);
bool is_full(struct Stack *stack);
struct Stack* create_stack(unsigned int capacity);

int main(int argc, char *argv[]){

  int N; // size of N x N board
  clock_t start, end;

  if(argc != 2){
    usage(argv[0]);
    exit(-1);
  }
  N = atoi(argv[1]);

  // allocating memory for the array
  int **board = malloc(N * sizeof(int *));
  for(int i = 0; i < N; i++){
    board[i] = malloc(N * sizeof(int));
  }
  initialize_board(board, N);

  start = clock();
  printf("Iterative implementation of N_queens\n\n");
  unsigned int stack_size = N;
  struct Stack *i_stack = create_stack(stack_size);
  struct Stack *j_stack = create_stack(stack_size);
  N_queens_it(board, N, i_stack, j_stack);
  printf("Solution to placing %d queens on a %d x %d board is:\n\n", N, N, N);
  print_solution(board, N);
  end = clock();

  double time = ((double)(end-start)/CLOCKS_PER_SEC);
  printf("\nRunning time for placing %d queens on a %d x %d board %lf seconds\n", N, N, N, time);

  free(board);
  return 0;
}
/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage: %s <size_of_board>\n", prog_name);
} /* usage */

// function which initializes the NxN board
void initialize_board(int **board, int n){
  int i, j;
  for(i = 0; i < n; i++){
    for(j =0; j < n; j++){
      board[i][j] = 0;
    }
  }
}
// function which prints the result
void print_solution(int **board, int n){
  int i, j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      printf(" %d ", board[i][j]);
    }
    printf("\n");
  }
}

/*
  Function which checks if it is safe to add a queen in this row and column
  - This function does a row check, a col check and a diagonal check.
*/
bool check_is_safe(int **board, int row, int col, int n){
  // checking the row
  int i, j;

  // this checks to see if there is any queen in the row
  for(i = 0; i < n; i++){
    if(board[row][i] == 1){
      return false;
    }
  }
  // this checks to see if there is any queen in the col
  for(i = 0; i < n; i++){
    if(board[i][col] == 1){
      return false;
    }
  }
  // checking upper diagonal, left side
  for(i = row, j = col; i >= 0 && j >= 0; i--, j--){
    if(board[i][j] == 1){
      return false;
    }
  }

  // checking lower diagonal, left side
  for(i = row, j = col; i < n && j >= 0; i++, j--){
    if(board[i][j] == 1){
      return false;
    }
  }
  // checking lower diagonal, right side
  for(i = row, j = col; i < n && j < n; i++, j++){
    if(board[i][j] == 1){
      return false;
    }
  }
  // checking upper diagonal right side
  for(i = row, j = col; i >= 0 && j < n; i--, j++){
    if(board[i][j] == 1){
      return false;
    }
  }

  /* we only check the left side in this function as the right side from col is
  not allocated yet.
  */
  return true;
}

/*
  Iterative NQueens function with stack and backtracking
*/

void N_queens_it(int **board, int n, struct Stack *i_stack, struct Stack *j_stack){

  int Q_placed = 0;
  int i = 0, j = 0;
  while(Q_placed != n){
    if(i < n && j < n){
      if(check_is_safe(board, i, j, n)){
        board[i][j] = 1;
        push(i_stack, i);
        push(j_stack, j);
        Q_placed++;
      }
    }
    // backtracking, if we are still in this loop and i == n we have not reached a solution
    if(i == n){
      // we need to parallelize this part of the code
      i = pop(i_stack);
      j = pop(j_stack);
      board[i][j] = 0;
      Q_placed--;
    }
    j++;
    if(j > (n-1)){
      i++;
      j = 0;
    }
  }
}

/*
  These functions enable the stack
*/
struct Stack* create_stack(unsigned int capacity){
  // this function allocates memory for the stack
  struct Stack *stack = (struct Stack*)malloc(sizeof(struct Stack));
  stack->capacity = capacity;
  stack->top = -1;
  stack->array = (int *) malloc(stack->capacity * sizeof(int));
  return stack;
}
bool is_full(struct Stack *stack){
  if(stack->top == stack->capacity - 1){
    return true;
  }else{
    return false;
  }
}
bool is_empty(struct Stack *stack){
  if(stack->top == -1){
    return true;
  }else{
    return false;
  }
}

void push(struct Stack *stack, int item){
  // adds an element onto the stack
  if(is_full(stack) == false){
    stack->array[++stack->top] = item;
  }
}
int pop(struct Stack *stack){
  // pops an element of the stack
  if(is_empty(stack) == false){
    return stack->array[stack->top--];
  }
  return -1;
}
