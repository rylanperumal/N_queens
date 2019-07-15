/*

* This program finds the number of solutions to the N queens problem.
* Assumes twos-complement architecture
*
* Usage: np N
*       Where N is the size of the board.
*
* This program will print the number of solutions.
*
*/

#include "shared.h"
#include "N_queens_cuda_unoptimized.h"
#include <cuda_runtime.h>
#include "common/inc/helper_cuda.h"
#include "common/inc/helper_functions.h"

int queen_count = 0;
int depth = DEPTH;
int threads_per_block = THREADS_PER_BLOCK;
bool verbose = false;
// bool cpu_only = false;
bool info_only = false;
int gpu_index = 0;


/* This function prints how the program is used and what the commands do */
void usage(const char *name){
    printf("Usage: %s [-i] [-g gpu_index] [-t threads_per_block] [-d depth] [-v] queen_count \n\n", name);
    printf("Options: \n");
    printf("\t-i Info mode. Displays information on the applied graphic cards. \n");
    printf("\t-g Index of the GPU device is used. \n");
    printf("\t-t Used to set the count of threads for each block [1...512]\n");
    printf("\t-d Indicates the depth to which the solutions shall be precalculated \n");
    printf("\t   Influences the number of threads that are used. [1...queen_count]\n");
    printf("\t-v Verbose mode. Displays additional info \n");
    // printf("\t-c CPU mode. No GPU involved \n")
}

/* This function checks the validity of the arguments that are passed in and exits the program if they are not with a message */
void check_arguments(){
    if (1 > threads_per_block || 512 < threads_per_block){
      printf("Threads per Block must be between %d and %d, inclusive. \n", 1, 512);
      exit(0);
    }

    if(MIN_BOARDSIZE > queen_count || MAX_BOARDSIZE < queen_count){
      printf("Queen count must be between %d and %d, inclusive. \n", MIN_BOARDSIZE, MAX_BOARDSIZE);
      exit(0);
    }

    if(1 > depth || queen_count < depth){
      printf("Depth must be between %d and %d, inclusive. \n", 1, queen_count);
      exit(0);
    }
}

/* This function  prints the parameters used in the main function*/
void print_arguments(){

  printf("Parameters: %d Queens / %d Threads per block / %d Depth / GPU %d /", queen_count, threads_per_block, depth, gpu_index);
  if(!verbose){
    printf("Not ");
  }
  printf("Verbose / ");
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

  if(strcmp(argv[*arg_index], "-g") == 0){
    (*arg_index)++;
    gpu_index = atoi(argv[*arg_index]);
    return;
  }

  if(strcmp(argv[*arg_index], "-v") == 0){
    verbose = true;
    return;
  }

  // if(strcmp(argv[*arg_index], "-c") == 0){
  //   cpu_only = true;
  //   return;
  //   }

  if(strcmp(argv[*arg_index], "-i") == 0){
    info_only = true;
    return;
    }

  if(*arg_index == argc - 1){
    queen_count = atoi(argv[*arg_index]);
  } else{
    usage(argv[0]);
  }
}

void parse_cmd_line(int argc, char *argv[]){
  if(argc < 2){
    usage(argv[0]);
  }

  for(int i = 1; i<argc; i++){
    parse_arguments(&i, argc, argv);
  }

  print_arguments();
  check_arguments();
}

/* This is a welcoming function go start the program */
void greetings(int argc, char** argv){
    printf("\n");
    printf("N Queens Problem in CUDA\n");
    printf("------------------------------------------------------------------\n");
    parse_cmd_line(argc, argv);
    printf("This program calculates the total number of solutions to the %d Queens problem.\n", queen_count);
    // cuda_info();
    printf("\n");
}


/* This function takes in the times and ouputs it in a logical format */
char * get_duration(ULL clock_difference, bool show_millis){
      int milliseconds = (int)(clock_difference % CLOCKS_PER_SEC);
      int seconds = (int)((clock_difference / CLOCKS_PER_SEC) % 60);
      int minutes = (int)((clock_difference / CLOCKS_PER_SEC*60ULL) % 60);
      int hours = (int)((clock_difference / CLOCKS_PER_SEC*60ULL*60) % 24);
      int days = (int)(clock_difference / CLOCKS_PER_SEC*60ULL*60*24);

      char results[20];

      if(show_millis){
        snprintf(results, 20, "%d:%02d:%02d:%02d:%03d", days, hours, minutes, seconds, milliseconds);
      } else{
        snprintf(results, 20, "%d:%02d:%02d:%02d", days, hours, minutes, seconds);
      }
      return strdup(results);
}

/* This is a helper function to get the time */
char *get_current_time(){
  time_t current_time;
  time(&current_time);

  char * result = ctime(&current_time);
  result[strlen(result)-1] = '\0';

  return result;
}

/* main routine for N Queens program.*/

int main(int argc, char** argv){

  ULL start, finish;

  greetings(argc, argv);

  // if(info_only){
  //   return 0;
  // }
  int devID = findCudaDevice(argc, (const char **) argv);
  printf("\n\n");

  printf("--[Started calculation at %s]-----------------\n", get_current_time());

  start = clock();
  ULL solution_count = calculate_solutions(gpu_index, queen_count, depth, threads_per_block, verbose);
  finish = clock();

  printf("--[Finished calculation at %s]----------------\n\n", get_current_time());
  printf("\n\n");

  printf("Solution: %llu\n", solution_count);
  printf("Duration: %s\n", get_duration(finish - start, true));

  return 0;
}
