#!/usr/bin/env python
#
# Toy program using mpi_fanout: prints squares of the first 200
# integers, "fanning out" the calculation over all MPI tasks.
#
# Usage: mpiexec -np 4 ./02-mpi-fanout-hello-world.py

import mpi_fanout

# This line must execute on master + workers, so that all cores "know" the definition of f.
def f(n):
    if n == 0:
        raise Exception('deal with it')
    return n**2

# Only returns on master.
mpi_fanout.init()

# "Fan out" 200 tasks over workers.  Each task computes one value of n^2.
task_list = [ mpi_fanout.task(f,n) for n in xrange(200) ]
return_values = mpi_fanout.run_tasks(task_list)

# The result is equivalent to just computing all n^2 values on the master.
assert return_values[1:] == [ n**2 for n in xrange(1,200,1) ]
assert return_values[0] is None
print 'Success!'

# Without putting this at the end, the program will hang!
mpi_fanout.exit()
