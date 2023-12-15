"""Multiprocessing wrapper functions."""
from contextlib import closing

from pathos.multiprocessing import ProcessPool
import multiprocessing

def process_map(fn, iterable, num_processes=4):
    """
    Apply a function to an iterable using multiple processes.

    Args:
        fn (callable): A function to apply to each element of the iterable.
        iterable (iterable): An iterable of inputs to the function.
        num_processes (int): The number of processes to use for parallelization.

    Returns:
        list: A list of results obtained by applying the function to each element of the iterable.
    """
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Apply the function to each element of the iterable using multiple processes
        results = pool.map(fn, iterable)
    return results, multiprocessing.cpu_count()

def process_starmap(fn, arg_tuples, num_processes=4):
    """
    Apply a function to multiple argument tuples using multiple processes.

    Args:
        fn (callable): A function to apply to each argument tuple.
        arg_tuples (iterable): An iterable of argument tuples to apply the function to.
        num_processes (int): The number of processes to use for parallelization.

    Returns:
        list: A list of results obtained by applying the function to each argument tuple.
    """
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Apply the function to each argument tuple using multiple processes
        results = pool.starmap(fn, arg_tuples)
    return results, multiprocessing.cpu_count()

def parallelize(num_processes=multiprocessing.cpu_count()):
    """
    Decorator that parallelizes a function or method using the pathos library.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create a pathos ProcessPool with the specified number of processes
            pool = ProcessPool(num_processes)

            print(f"Multiprocessing on {num_processes} CPUs")

            try:
                # Use the ProcessPool to run the function or method with the given arguments
                if hasattr(func, "__self__") and hasattr(func, "__func__"):
                    # It's a method, so invoke it on the object using pool.apipe
                    result = pool.apipe(func, *args, **kwargs).get()
                else:
                    # It's a regular function, so use pool.apipe
                    result = pool.apipe(func, *args, **kwargs).get()
            finally:
                # Close the pool to release resources
                pool.close()
                pool.join()

            return result, multiprocessing.cpu_count()

        return wrapper

    return decorator
