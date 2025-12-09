# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import multiprocessing
import subprocess
import logging
import signal
import sys
import shlex
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass


process_utils_logger = logging.getLogger("process_utils")


@dataclass
class WorkerContext:
    worker_id: int
    device_id: str


class WorkerContextManager:
    """
    Provides WorkerContext (worker_id, device_id) to worker processes.

    In multiprocessing mode:
        - Each worker process runs initializer(), which pulls one context from
        the shared queue and stores it in process-local `_local_context`.

    In single-process (baseline) mode:
        - The caller manually sets the context via WorkerContextManager.set().

    Worker functions then retrieve the context uniformly via WorkerContextManager.get().
    """

    _local_context: Optional[WorkerContext] = None

    def __init__(self, device_ids: list[str]):
        self.queue = multiprocessing.Manager().Queue()
        for worker_id, device_id in enumerate(device_ids):
            self.queue.put(WorkerContext(worker_id, device_id))

    @staticmethod
    def get() -> Optional[WorkerContext]:
        return WorkerContextManager._local_context

    @staticmethod
    def set(context: WorkerContext):
        WorkerContextManager._local_context = context

    def initializer(self):
        """Used inside multiprocessing.Pool to set per-worker context."""
        ctx = self.queue.get()
        WorkerContextManager.set(ctx)


class MultiprocessExecutor:
    """
    Wrapper of multiprocessing + progress bar + time budget.

    Example:
        executor = MultiprocessExecutor(num_workers=4)
        executor.run(task_list, worker_func)
    """

    def __init__(
        self,
        num_workers: int,
        initializer=None,
        initializer_inputs=(),
        time_budget=None,
    ):
        self.num_workers = num_workers
        self.initializer = initializer
        self.initializer_inputs = initializer_inputs
        self.time_budget = time_budget

    def run(self, task_list, worker_fn):
        results = []

        # Create a multiprocessing pool.
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        with multiprocessing.Pool(
            self.num_workers, self.initializer, self.initializer_inputs
        ) as worker_pool:
            signal.signal(signal.SIGINT, sigint_handler)
            # Use tqdm to create a progress bar.
            with tqdm(total=len(task_list)) as pbar:
                try:
                    # Use imap_unordered to asynchronously execute the worker function on each task.
                    for result in worker_pool.imap_unordered(worker_fn, task_list):
                        results.append(result)
                        pbar.update(1)  # Update progress bar.
                        # If time limit is reached, stop progress wrapper.
                        if self.time_budget is not None and self.time_budget.expired():
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
    """
    Wrapper around subprocess.run() with optional timeout and error handling.

    Example:
        result = process_utils.run_command(
            process_utils.RunPack(
                command=["echo", "hello"],
                check=True,
            )
        )
    """
    command = run_pack.command
    check = run_pack.check
    timeout_seconds = run_pack.timeout_seconds

    result = None
    is_timeout = False
    try:
        # Convert the command list to a command string for logging.
        command_str = shlex.join(command)
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
            f"Command '{command_str}' failed with exit code {e.returncode}.\n"
            f"stderr:\n{e.stderr}"
        )
        if check:
            raise
    except KeyboardInterrupt:
        print("Ctrl+C detected, terminating child processes...")

    return RunResult(result, is_timeout)
