#!/usr/bin/env python3
#
# Toy program using mpi_fanout: prints squares of the first 200
# integers, "fanning out" the calculation over all MPI tasks.
#
# Usage: mpiexec -np 4 ./02-mpi-fanout-hello-world.py

import mpi_fanout

# This line must execute on master + workers, so that all cores "know" the definition of f.
def f(n):
    return n**2

# Only returns on master.
mpi_fanout.init()

# "Fan out" 200 tasks over workers.  Each task computes one value of n^2.
task_list = [ mpi_fanout.task(f,n) for n in range(200) ]
return_values = mpi_fanout.run_tasks(task_list)

# The result is equivalent to just computing all n^2 values on the master.
assert return_values == [ n**2 for n in range(200) ]
print('Success!')

# Without putting this at the end, the program will hang!
mpi_fanout.exit()
