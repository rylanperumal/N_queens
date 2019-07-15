#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

/*
  Function definitions
*/
void usage(char prog_name[]);
void initialize_board(int ** board, int n);
void print_solution(int **board, int n);
bool check_is_safe(int **board, int row, int col, int n);
bool N_queens(int **board, int col, int n);

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
  if(N_queens(board, 0, N) == false){
    printf("Solution does not exist\n\n");
  }else{
    printf("Solution to placing %d queens on a %d x %d board is:\n\n", N, N, N);
    print_solution(board, N);
  }
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
  for(i = 0; i < col; i++){
    if(board[row][i] == 1){
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
  for(i = row, j = col; i < n && j <= 0; i++, j--){
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
  Nqueens recursive function with backtracking

*/

bool N_queens(int **board, int col, int n){

  // if there are N queens are placed return true
  if(col >= n){
    return true;
  }

  // consider column and try placing a queen in all the rows one by one
  for(int i = 0; i < n; i++){
    // check if placement is safe
    if(check_is_safe(board, i, col, n)){
      board[i][col] = 1;
      if(N_queens(board, col+1, n)){
        return true;
      }

      // if placing queen at row i and col col, then remove the queen i.e backtrack
      board[i][col] = 0;
    }
  }
  // if queen cannot be placed in any row in this column
  return false;
}
