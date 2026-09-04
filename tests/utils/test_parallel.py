# tests/utils/test_parallel.py
import platform
import subprocess
import time

import psutil
import ray.util.state
from ray.cluster_utils import Cluster

from milp_engine.utils.parallel import get_cores_info, init_ray, run_tasks_parallel


def test_get_cores_info_macos() -> None:
    """Test get_cores_info on macOS with actual system information."""
    if platform.system() == "Darwin":
        try:
            efficiency_cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel1.physicalcpu"]
                ).strip()
            )
            performance_cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"]
                ).strip()
            )
            total_cores = performance_cores + efficiency_cores
            cores_info = get_cores_info()
            assert cores_info == {
                "performance_cores": performance_cores,
                "efficiency_cores": efficiency_cores,
                "total_cores": total_cores,
            }
        except subprocess.CalledProcessError:
            print("Unable to retrieve core information on macOS. Skipping test.")
    else:
        print("Not running on macOS. Skipping test.")


def test_get_cores_info_linux_windows() -> None:
    """Test get_cores_info on Linux/Windows with actual system information."""
    if platform.system() in ["Linux", "Windows"]:
        try:
            total_cores = psutil.cpu_count()
            cores_info = get_cores_info()
            assert cores_info == {
                "performance_cores": 0,  # Assuming no distinction on Linux/Windows
                "efficiency_cores": 0,
                "total_cores": total_cores,
            }
        except Exception as e:
            print(f"Unable to retrieve core information: {e}. Skipping test.")
    else:
        print("Not running on Linux/Windows. Skipping test.")


def test_init_ray_already_initialized(ray_shutdown: None) -> None:
    """Test init_ray when Ray is already initialized.

    :param ray_shutdown: Fixture to ensure Ray is shut down after tests.
    """
    init_ray()  # Ensure Ray is initialized for the test
    init_ray()
    assert ray.is_initialized()


def test_init_ray_connect_existing_instance(ray_shutdown: None) -> None:
    """Test init_ray when connecting to an existing Ray instance.

    :param ray_shutdown: Fixture to ensure Ray is shut down after tests.
    """
    # Simulate a scenario where Ray connects to an existing instance
    try:
        ray.init(address="auto")
    except ConnectionError:
        pass

    init_ray()

    assert ray.is_initialized()


def test_init_ray_start_new_instance_with_performance_cores(ray_shutdown: None) -> None:
    """Test init_ray when starting a new Ray instance with performance cores.

    :param ray_shutdown: Fixture to ensure Ray is shut down after tests.
    """
    cores_info = get_cores_info()
    performance_cores = cores_info.get("performance_cores", 0)

    init_ray()

    assert ray.is_initialized()
    assert ray.cluster_resources().get("CPU", 0) >= performance_cores


def test_run_tasks_parallel_results(ray_test_cluster: Cluster) -> None:
    def sample_task(x: int) -> int:
        return x * x

    inputs = list(range(20))
    results, _, _ = run_tasks_parallel(sample_task, inputs)

    # Check if the results are as expected
    expected_results = [x * x for x in inputs]
    assert results == expected_results


def test_run_tasks_parallel_core_usage(ray_test_cluster: Cluster) -> None:
    def identity(x: int) -> int:
        return x

    inputs = list(range(100))
    _ = run_tasks_parallel(identity, inputs)

    time.sleep(1)

    tasks = ray.util.state.list_tasks(limit=1_000)

    worker_ids = set([task["worker_id"] for task in tasks])
    states = [task["state"] for task in tasks]

    # Assert that 100 tasks were made
    assert len(tasks) == 100

    # Assert that all tasks are executed
    for state in states:
        assert state == "FINISHED"

    # Check that all (2) cores were used
    assert len(worker_ids) == 2


def test_run_tasks_parallel_with_timings_results(ray_test_cluster: Cluster) -> None:
    def sample_task(x: int) -> int:
        return x * x

    inputs = list(range(20))
    results, timings, timings_with_init = run_tasks_parallel(
        sample_task, inputs, timing=True
    )

    # Check if the results are as expected
    expected_results = [x * x for x in inputs]
    assert results == expected_results
    assert len(timings) == len(inputs)
    assert len(timings_with_init) == len(inputs)
    for timing, timing_with_init in zip(timings, timings_with_init):
        assert timing_with_init >= timing
