# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import csv
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from amdsharktuner import common, process_utils


# List of tested ROCm architectures.
ROCM_ARCHITECTURES = ["gfx942", "gfx950", "gfx1100", "gfx1201"]


@dataclass
class ConvToIgemmInfo:
    """
    Stores information about convolution to IGEMM transformation.

    Corresponds to ConvToIgemmInfo struct in IREE:
    https://github.com/iree-org/iree/blob/d3440737cc56a4d1b20c72181d9a37f194bd3ce5/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L373-L379

    Note: convolution_dims is not included here because this struct is IGEMM-specific,
    while convolution_dims is needed by both IGEMM and direct convolution strategies.
    It's stored in ROCmConvolutionOpInfo instead.
    """

    is_batch_dim_last: bool = False
    is_spatial_dim_last: bool = False
    conv_to_igemm_dim: dict[int, int] = field(default_factory=dict)
    input_channel_dim_to_size: dict[int, int] = field(default_factory=dict)


@dataclass
class RocProfConfig(common.BenchmarkToolConfig):
    """Configuration for rocprof-based benchmarking."""

    benchmark_fn: Callable
    iree_benchmark_module_flags: list[str]
    rocprof_output_dir: Path
    rocprof_output_filename_prefix: str
    rocprof_output_format: str


def compute_rocprof_avg_kernel_time(trace_rows: list[dict]) -> float:
    """
    Compute average kernel execution time from rocprof trace data.

    Args:
        trace_rows: List of dictionaries containing rocprof trace data with
                   Kernel_Name, Start_Timestamp, and End_Timestamp fields.

    Returns:
        Average kernel execution time in microseconds.

    Raises:
        ValueError: If trace is empty or missing required columns.
        RuntimeError: If initializer dispatch was measured instead of main kernel.
    """
    if not trace_rows:
        raise ValueError("Rocprof kernel trace is empty.")

    required_cols = {"Kernel_Name", "Start_Timestamp", "End_Timestamp"}
    # Only need to check the first row.
    row_keys = set(trace_rows[0].keys())
    missing = required_cols - row_keys
    if missing:
        raise ValueError(
            f"Missing required columns in rocprof kernel trace snippet rows: {sorted(missing)}"
        )

    # Skip warm-up iterations.
    if len(trace_rows) >= 20:
        trace_rows = trace_rows[10:]  # Drop first 10 rows.
    else:
        logging.warning(
            "Rocprof kernel trace CSV contains insufficient records; timing results may be unreliable or noisy."
        )

    init_dispatch_fn_name_key = "_buffer"
    if any(init_dispatch_fn_name_key in str(row["Kernel_Name"]) for row in trace_rows):
        raise RuntimeError(
            "Rocprof measured the initializer dispatch instead of the main kernel computation."
        )

    clk_diffs_ns = []
    for row in trace_rows:
        start = float(row["Start_Timestamp"])
        end = float(row["End_Timestamp"])
        clk_diffs_ns.append(end - start)

    avg_clk_ns = sum(clk_diffs_ns) / len(clk_diffs_ns)
    avg_clk_us = avg_clk_ns / 1000.0

    return avg_clk_us


@dataclass
class RocProfBenchmarkResult:
    """
    Benchmark result from rocprof kernel timing.

    Must provide the same interface as libtuner's BenchmarkResult for compatibility. See issue
    https://github.com/nod-ai/amd-shark-ai/issues/2908 for more details.
    """

    candidate_id: int
    time: float
    device_id: str

    def is_valid(self) -> bool:
        return math.isfinite(self.time)

    def __iter__(self):
        return iter((self.candidate_id, self.time, self.device_id))


def run_rocprof_command(benchmark_pack: Any) -> RocProfBenchmarkResult:
    """
    Run benchmark using rocprof for kernel timing.

    Args:
        benchmark_pack: Benchmark configuration and candidate information.

    Returns:
        RocProfBenchmarkResult with the measured kernel time.
    """
    assert isinstance(benchmark_pack.benchmark_tool_config, RocProfConfig)
    benchmark_tool_config: RocProfConfig = benchmark_pack.benchmark_tool_config
    candidate_tracker = benchmark_pack.candidate_tracker

    candidate_id = candidate_tracker.candidate_id
    vmfb_path = candidate_tracker.compiled_vmfb_path
    worker_ctx = process_utils.WorkerContextManager.get()
    assert (
        worker_ctx is not None
    ), "Missing WorkerContext. Did you forget to set it in baseline?"
    device_id = worker_ctx.device_id

    output_file = f"{benchmark_tool_config.rocprof_output_dir}/{candidate_id}"
    rocprof_command = [
        "rocprofv3",
        "--kernel-trace",
        f"--output-file={output_file}",
        f"--output-format={benchmark_tool_config.rocprof_output_format}",
    ]
    benchmark_command = [
        "iree-benchmark-module",
        f"--module={vmfb_path}",
        f"--device={device_id}",
    ]
    benchmark_command += (
        benchmark_pack.benchmark_tool_config.iree_benchmark_module_flags
    )
    measure_cmd = rocprof_command + ["--"] + benchmark_command

    result = process_utils.run_command(
        process_utils.RunPack(
            command=measure_cmd,
            timeout_seconds=benchmark_pack.benchmark_timeout,
        )
    )

    if result.is_timeout:
        return RocProfBenchmarkResult(
            candidate_id=candidate_id,
            time=math.inf,
            device_id=str(device_id),
        )

    trace_path = Path(
        f"{output_file}{benchmark_tool_config.rocprof_output_filename_prefix}.{benchmark_tool_config.rocprof_output_format}"
    )
    benchmark_pack.candidate_tracker.kernel_trace_path = trace_path
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"File not found: {trace_path}")
    with open(trace_path, newline="") as f:
        trace_reader = csv.DictReader(f)
        trace_rows = list(trace_reader)

    time = compute_rocprof_avg_kernel_time(trace_rows)
    logging.debug(f"Rocprof benchmark time of candidate {candidate_id}: {time:.2f} us")
    return RocProfBenchmarkResult(
        candidate_id=candidate_id,
        time=time,
        device_id=str(device_id),
    )
