Author: Tasneem Abed and Rylan Perumal

N-Queens Problem using MPI

How to run:
$ make clean
$ make
For the run files either of the following commands will work depending on how
many nodes you want use.
$ ./run_single_node.sh
$ ./run_two_nodes.sh
$ ./run_four_nodes.sh
$ ./run_eight_nodes.sh

NOTE: Our implementation requires the number of processes to be N + 1, where N is
the board size.

NOTE: The run.sh only tests for values of N from 5 to 16. N > 16 takes some
time to run.

To add the test for N = 17, add these lines to each of the run.sh files

1 host: mpiexec -n 18 ./mpi 17 |& tee -a out_one_node.txt;
2 hosts: mpiexec -hosts mscluster0,mscluster1 -n 18 ./mpi 17 |& tee -a out_two_node.txt;
4 hosts: mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3 -n 18 ./mpi 17 |& tee -a out_four_node.txt;
8 hosts: mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 18 ./mpi 17 |& tee -a out_eight_node.txt;

To add the test for N = 18, add these lines to each of the run.sh files

1 host: mpiexec -n 19 ./mpi 18 |& tee -a out_one_node.txt;
2 hosts: mpiexec -hosts mscluster0,mscluster1 -n 19 ./mpi 18 |& tee -a out_two_node.txt;
4 hosts: mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3 -n 19 ./mpi 18 |& tee -a out_four_node.txt;
8 hosts: mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 19 ./mpi 18 |& tee -a out_eight_node.txt;


NOTE: To run on two, four or eight nodes, the run.sh files are only compatible with
the MSL cluster which is at the University of the Witwatersrand. However they can
easily be modified to run across any cluster.
