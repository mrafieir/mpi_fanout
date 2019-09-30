"""
mpi_fanout: low-budget library for MPI programs which run entirely in a "master-worker" mode,
where the master core can "fan out" calculations over the worker cores, and the worker cores
don't do anything else.

  - Early in the program, call mpi_fanout.init() on all cores.  On the master core (i.e. MPI rank 0),
    init() will return immediately.  On worker cores (MPI rank > 0), it will never return!

  - From this point on, the program is effectively serial code running on the master core, except 
    that the master core can create 'tasks' for the worker cores.  A task is a deferred function
    evaluation.  The arguments are prepared on the master and sent to the worker.  The function call
    is performed on the worker, and the return value is sent back to the master.

    Here is a toy example that computes n^2 for n=0,...,199, but "fanning out" the calculation
    over all cores:

       # This line must execute on master + workers, so that all cores "know" the definition of f.
       def f(n): return n**2

       # Only returns on master.
       mpi_fanout.init()

       # "Fan out" 200 tasks over workers.  Each task computes one value of n^2.
       task_list = [ mpi_fanout.task(f,n) for n in range(200) ]
       return_values = mpi_fanout.run_tasks(task_list)

       # The result is equivalent to just computing all n^2 values on the master.
       assert return_values == [ n**2 for n in range(200) ]

       # Without putting this at the end, the program will hang!
       mpi_fanout.exit()

  - At the end of the program, you should call mpi_fanout.exit(), or the program will hang.  (This
    is a problem on clusters, since the hung program will monopolize its nodes until it is either
    manually killed or its wallclock limit is reached!)

TODO: currently output from all cores is written to stdout, which results in interleaving.
It would be better to multiplex output over log files.
"""

import sys
import pickle
from mpi4py import MPI

init_called = False
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()


class task:
    """
    A 'task' represents a deferred function evaluation func(*args, **kwds)
    which is prepared by the master core, and will be evaluated by a worker.

    Note that the function object and its arguments must be pickleable, or
    you'll get a cryptic error message!  (e.g. you can't use a lambda-function).
    """
    
    def __init__(self, func, *args, **kwds):
        if not callable(func):
            raise RuntimeError("mpi_fanout.task constructor: 'func' argument must be a function (or callable object)")

        self.func = func
        self.args = args
        self.kwds = kwds
        

def init(silent=False):
    """
    init() should be called once per program, on all (master + worker) cores.

    On the master core (MPI task 0) it returns immediately.  On worker cores,
    it never returns!  Instead, the workers enter a loop where they are waiting
    for tasks from the master.
    """
    
    global init_called, mpi_rank
    
    if init_called:
        raise RuntimeError("double call to mpi_fanout.init()")

    init_called = True

    if mpi_rank == 0:
        if not silent:
            print(f'mpi_fanout: number of MPI tasks = {mpi_size}')
        return

    # Main worker loop
    while True:
        my_tasks = MPI.COMM_WORLD.scatter(None, root=0)

        if my_tasks is None:
            MPI.Finalize()
            sys.exit(0)
            
        my_rvals = _process_tasks(my_tasks)
        MPI.COMM_WORLD.gather(my_rvals, root=0)


def run_tasks(task_list):
    """
    Given a list of 'task' objects representing deferred function evaluations,
    the function calls are "fanned out" over the workers, and the return values
    are gathered into a single list, which is returned on the master.
    
    In other words, run_tasks() is equivalent to:
    
       return [ t.func(*t.args, **t.kwds) for t in task_list ]

    but the function evaluations are "fanned out" over all cores.
    """
    
    global init_called, mpi_rank, mpi_size

    if not init_called:
        raise RuntimeError("mpi_fanout.run_tasks() called without prior call to mpi_fanout.init()")

    task_list = list(task_list)  # coerce to list
    assert all(isinstance(t,task) for t in task_list)
    assert mpi_rank == 0
    
    # "Unflatten" 1-d task_list to 2-d list-of-lists (where outer index is MPI task, inner index is task)
    tasks_2d = [task_list[i::mpi_size] for i in range(mpi_size)]
    
    my_tasks = MPI.COMM_WORLD.scatter(tasks_2d, root=0)
    my_rvals = _process_tasks(my_tasks)
    rvals_2d = MPI.COMM_WORLD.gather(my_rvals, root=0)

    # Flatten rvals_2d (list of lists) to a 1-d list of return values.
    return_values = [ rvals_2d[i%mpi_size][i//mpi_size] for i in range(len(task_list)) ]
    return return_values


def _process_tasks(my_task_list):
    return [ t.func(*t.args, **t.kwds) for t in my_task_list ]


def exit():
    """
    Should be called at the end of the program, on the master core.  Otherwise, the
    program will hang!  This is a problem on clusters, since the hung program will
    monopolize its nodes until it is either manually killed or its wallclock limit
    is reached.
    """
    
    global init_called, mpi_rank, mpi_size
    
    if not init_called:
        raise RuntimeError("mpi_fanout.end_program() called without prior call to mpi_fanout.init()")

    assert mpi_rank == 0

    MPI.COMM_WORLD.scatter([None]*mpi_size, root=0)
    MPI.Finalize()
    sys.exit(0)

