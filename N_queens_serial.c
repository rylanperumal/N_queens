#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/*
  Unncomment line 9 below to print solution
*/
// #define PRINT

int total_solutions = 0;

// checks if queens attack each other
bool check_attack(int board[], int col){
  for(int i = 0; i < col; i++){
    /*
      This checks to see if there is an already placed queen which blocks
      the current queen being placed. We only check the row position here since
      we add queens column wise.
    */
    if(board[i] == board[col]){
      return false;
    }
    /*
      This checks to see if the queen being placed is in diagonal conflict
      with any other queen already on the board.
    */
    if((abs(board[i] - board[col])) == (col - i)){
      return false;
    }
  }
  // function returns if there are no conflicts
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
    // 1D solution into 2D
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

  if(col >= n){ // we have reached a solution
    total_solutions++;

    // prints out the first solution
    if(total_solutions == 1){
      // allocating memory for the array
      int *solution = (int *)malloc(n * n * sizeof(int));
      initialize_board(solution, n);
      // here we want to print out the solution in 2D board format
      print_solution(solution, board, n);
    }
    // prints out all 1D solutions
    #ifdef PRINT
    printf("\n");
    for(i = 0; i < n; i++){
      printf(" %d ", board[i]);
    }
    printf("\n");
    #endif

  }else{

    for(i = 0; i < n; i++){
      // trying to place all values from 1 to N in the column
      board[col] = i;
      // this checks to see if the value placed in the column is valid
      if(check_attack(board, col)){
        // here we go the the next column if the current column value is valid
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
