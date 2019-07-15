Author: Tasneem Abed and Rylan Perumal

N-Queens Problem in Serial

How to run:
$ make clean
$ make
$ ./run.sh

NOTE: The run.sh only tests for values of N, from 5 to 16. N > 16 takes some
time to run.

To add the test of 17 and 18, add these lines to the run.sh
./serial_tot 17 |& tee -a out_serial.txt;
./serial_tot 18 |& tee -a out_serial.txt;
