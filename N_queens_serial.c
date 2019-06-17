#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
// #define PRINT


int total_solutions = 0;

// checks if queens attack each other
bool no_conflicts(int board[], int col){
  for(int i = 0; i < col; i++){
    if(board[i] == board[col]){
      // checks the column
      return false;
    }
    if((abs(board[i] - board[col])) == (col - i)){
      // checks the diagonal
      return false;
    }
  }
  return true;
}
// initializes the board to 0's
void initialize_board(int *solution, int n){
  int i, j;
  for(i = 0; i < n; i++){
    for(j =0; j < n; j++){
      solution[i * n + j] = 0;
    }
  }
}
// function which prints the result
void print_solution(int *solution, int row_board[], int n){
  int i, j;
  // fill in the board
  for(j = 0; j < n; j++){
    solution[row_board[j] * n + j] = 1;
  }
  // prints the board
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      printf(" %d ", solution[i * n + j]);
    }
    printf("\n");
  }
}
void n_queens(int board[], int col, int n){
  int i;
  if(col >= n){
    total_solutions++;

    if(total_solutions == 1){
      // allocating memory for the array
      int *solution = (int *)malloc(n * n * sizeof(int));
      initialize_board(solution, n);
      // here we want to print out the solution in 2D board format
      print_solution(solution, board, n);
    }
    #ifdef PRINT
    printf("\n");
    for(i = 0; i < n; i++){
      printf(" %d ", board[i]);
    }
    printf("\n");
    #endif
  }else{
    for(i = 0; i < n; i++){
      board[col] = i;
      if(no_conflicts(board, col)){
        // here we do the recursion, if the position is valid
        n_queens(board, col + 1, n);
      }
    }
  }
}

/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage: %s <size_of_board>\n", prog_name);
} /* usage */

/*
  Main
*/
int main(int argc, char *argv[]){
  int N; // size of N x N board
  clock_t start, end;

  if(argc != 2){
    usage(argv[0]);
    exit(-1);
  }
  N = atoi(argv[1]);
  int board[N];
  start = clock();
  n_queens(board, 0, N);
  end = clock();
  printf("\nTotal number of solutions: %d\n", total_solutions);
  double time = ((double)(end-start)/CLOCKS_PER_SEC);
  printf("\nRunning time for placing %d queens on a %d x %d board %lf seconds\n", N, N, N, time);

  // free(board);
  return 0;
}
