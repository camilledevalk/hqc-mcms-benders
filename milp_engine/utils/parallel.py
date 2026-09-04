# milp_engine/utils/ray_utils.py
"""This module contains utility functions for Ray parallel processing."""
import platform
import subprocess
import time
from typing import Callable, Dict, List, Tuple, TypeVar

import psutil
import ray

TI = TypeVar("TI")
TO = TypeVar("TO")


def timed_task(task: Callable[[TI], TO]) -> Callable[[TI], Tuple[TO, float]]:
    """
    Wrap a function to measure its execution time.

    :param task: The function to wrap. It should take a single
        input and return a result.

    :returns:
        A new function that returns a tuple of:
        - result: The output of the original task.
        - duration: Execution time in seconds.
    """

    def wrapped(input: TI) -> Tuple[TO, float]:
        """
        Execute the task on a single input and measure execution time.

        :param input: Input value to pass to the task function.

        :returns:
            A tuple containing:
            - result: The output produced by the task.
            - duration: Time in seconds taken to execute the task.
        """
        start = time.perf_counter()
        result = task(input)
        end = time.perf_counter()
        return result, end - start

    return wrapped


def run_tasks_parallel(
    task: Callable[[TI], TO], inputs: List[TI], timing: bool = False
) -> Tuple[List[TO], List[float], List[float]]:
    """
    Run a function multiple times in parallel with different inputs using Ray.

    :param task: Function to execute.
    :param inputs: List of inputs to apply the function on.
    :param timing: If True, collect and return timing data.
    :return: Tuple of results, timings, and timings including Ray init time.
    """
    ray_init_time = 0.0
    if not ray.is_initialized():
        if timing:
            init_start = time.perf_counter()
            init_ray()
            init_end = time.perf_counter()
            ray_init_time = init_end - init_start
        else:
            init_ray()

    task_to_run = timed_task(task) if timing else task
    task_remote = ray.remote(task_to_run)

    futures = [task_remote.remote(input) for input in inputs]
    results_raw = ray.get(futures)

    if timing:
        results, durations = zip(*results_raw)
        timings_with_init = [t + ray_init_time for t in durations]
        return (list(results), list(durations), timings_with_init)
    else:
        return results_raw, [], []


def init_ray() -> None:
    """Initialize Ray for parallel processing."""
    # Check if Ray is already initialized
    if ray.is_initialized():
        print("Ray is already initialized.")
        return
    print("Initializing Ray...")
    try:
        # Attempt to connect to an existing Ray instance
        ray.init(address="auto", runtime_env={"working_dir": "."})
        print("Connected to an existing Ray instance.")
    except ConnectionError:
        # Start a new Ray instance if no existing instance is found
        cores_info = get_cores_info()
        if "error" in cores_info:
            print(cores_info["error"])
            return
        num_cpus = (
            cores_info["performance_cores"]
            if int(cores_info["performance_cores"]) > 0
            else cores_info["total_cores"]
        )
        ray.init(num_cpus=num_cpus)
        print(f"Started a new Ray instance with {num_cpus} cpus.")


def get_cores_info() -> Dict[str, int | str]:
    """Retrieve information about the CPU cores on the current system.

    For macOS:
        - Attempts to retrieve performance and efficiency cores using `sysctl`.
        - Returns a dictionary with counts of performance, efficiency, and total cores.
        - If retrieval fails, an error message is returned.

    For Linux and Windows:
        - Retrieves the total number of physical cores using `psutil`.
        - Performance and efficiency core counts are set to 0 on these platforms.
        - Returns a dictionary with the total number of cores.

    :returns: A dictionary with the following keys:
        -performance_cores: Number of performance cores (macOS only, 0 otherwise).
        -efficiency_cores: Number of efficiency cores (macOS only, 0 otherwise).
        -total_cores: Total number of physical cores.
        -error: Error message if retrieval fails, empty string otherwise.
    """
    system = platform.system()
    if system == "Darwin":  # macOS
        try:
            performance_cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True
                ).strip()
            )
            efficiency_cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel1.physicalcpu"], text=True
                ).strip()
            )
            return {
                "performance_cores": performance_cores,
                "efficiency_cores": efficiency_cores,
                "total_cores": performance_cores + efficiency_cores,
            }
        except subprocess.CalledProcessError:
            return {"error": "Unable to retrieve core information on macOS"}
    else:  # Linux and Windows
        try:
            total_cores = psutil.cpu_count(logical=False)
            return {
                "performance_cores": 0,  # Not available on Linux/Windows
                "efficiency_cores": 0,  # Not available on Linux/Windows
                "total_cores": total_cores,
            }
        except Exception as e:
            return {"error": f"Unable to retrieve core information: {str(e)}"}
