#!/usr/bin/env python3
#
# This toy program tests that the standard python library 'mpi4py'
# is working.  (It doesn't use mpi_fanout.)
#
# Usage: mpiexec -np 4 ./01-mpi4py-hello-world.py

import sys
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

sys.stdout.write(f'hello world {rank+1}/{size}\n')
