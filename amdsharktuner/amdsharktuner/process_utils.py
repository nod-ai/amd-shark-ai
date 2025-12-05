import multiprocessing
import queue
import subprocess
import logging
import signal
from tqdm import tqdm
from typing import Type, Optional, Callable, Iterable, Any
from dataclasses import dataclass, field

from . import common

# Declare global variables at the module level for multiprocessing.
worker_id = None
device_id = None

process_utils_logger = logging.getLogger("process_utils")

def init_worker_context(queue: multiprocessing.Queue) -> None:
    """Assign a static index to current process as the worker ordinal, and specify the device indices to be used"""
    global worker_id, device_id

    worker_id, device_id = queue.get()


def create_worker_context_queue(device_ids: list[str]) -> queue.Queue[tuple[int, int]]:
    """Create queue contains Worker ID and Device ID for worker initialization"""
    worker_contexts_queue = multiprocessing.Manager().Queue()
    for worker_id, device_id in enumerate(device_ids):
        worker_contexts_queue.put((worker_id, device_id))

    return worker_contexts_queue


def set_global_worker_id(worker_id_: int):
    global worker_id
    worker_id = worker_id_


def set_global_device_id(device_id_: int):
    global device_id
    device_id = device_id_


def get_global_worker_id():
    return worker_id


def get_global_device_id():
    return device_id

def multiprocess_progress_wrapper(
    num_worker: int,
    task_list: list,
    function: Callable,
    initializer: Optional[Callable] = None,
    initializer_inputs: Optional[Iterable[Any]] = None,
    time_budget: Optional[common.TimeBudget] = None,
) -> list[Any]:
    """Wrapper of multiprocessing pool and progress bar"""
    results = []
    initializer_inputs = initializer_inputs or ()

    # Create a multiprocessing pool.
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with multiprocessing.Pool(
        num_worker, initializer, initializer_inputs
    ) as worker_pool:
        signal.signal(signal.SIGINT, sigint_handler)
        # Use tqdm to create a progress bar.
        with tqdm(total=len(task_list)) as pbar:
            try:
                # Use imap_unordered to asynchronously execute the worker function on each task.
                for result in worker_pool.imap_unordered(function, task_list):
                    results.append(result)
                    pbar.update(1)  # Update progress bar.
                    # If time limit is reached, stop progress wrapper.
                    if time_budget is not None and time_budget.expired():
                        logging.warning(
                            f"Time limit reached, total {len(results)} results collected"
                        )
                        worker_pool.terminate()
                        worker_pool.join()
                        return results
            except KeyboardInterrupt:
                # If Ctrl+C is pressed, terminate all child processes.
                worker_pool.terminate()
                worker_pool.join()
                sys.exit(1)  # Exit the script.

    return results


@dataclass
class RunPack:
    command: list[str]
    check: bool = True
    timeout_seconds: Optional[float] = None


@dataclass
class RunResult:
    process_res: Optional[subprocess.CompletedProcess]
    is_timeout: bool


def run_command(run_pack: RunPack) -> RunResult:
    command = run_pack.command
    check = run_pack.check
    timeout_seconds = run_pack.timeout_seconds

    result = None
    is_timeout = False
    try:
        # Convert the command list to a command string for logging.
        command_str = " ".join(command)
        logging.debug(f"Run: {command_str}")

        # Add timeout to subprocess.run call.
        result = subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        logging.warning(
            f"Command '{command_str}' timed out after {timeout_seconds} seconds."
        )
        is_timeout = True
    except subprocess.CalledProcessError as e:
        print(e.output)
        logging.error(
            f"Command '{command_str}' returned non-zero exit status {e.returncode}."
        )
        logging.error(f"Command '{command_str}' failed with error: {e.stderr}")
        if check:
            raise
    except KeyboardInterrupt:
        print("Ctrl+C detected, terminating child processes...")

    return RunResult(result, is_timeout)