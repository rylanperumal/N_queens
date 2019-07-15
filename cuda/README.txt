Author: Tasneem Abed and Rylan Perumal

N-Queens Problem using CUDA

How to run:
$ make clean
$ make
The following commands will work depending on how
many threads or depth you want to use. The first number is the number threads to be used,
the second number is the depth.
$ ./run_32_2.sh
$ ./run_32_4.sh
$ ./run_32_5.sh
$ ./run_64_2.sh
$ ./run_64_4.sh
$ ./run_32_5.sh
$ ./run_128_2.sh
$ ./run_128_4.sh
$ ./run_128_5sh


If you want to run the program without the run file:
./cuda -t <number_of_threads_to_be_used> -d <depth_to_solve_in_serial> <N>

NOTE: The number of threads must be between 1 and 512. The depth must be at least
2 less than the specified N. eg if N = 10, d < 9.

Example run for 14 x 14 board, using 64 threads and at depth 2:
./cuda -t 64 -d 2 14
