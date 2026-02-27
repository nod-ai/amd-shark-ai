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
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu, linalg  # type: ignore

from amdsharktuner import common, process_utils


# The Key name for the 'amdgpu-waves-per-eu' within the llvm_func_attrs attribute.
WAVES_PER_EU_KEY = "amdgpu-waves-per-eu"

# List of tested ROCm architectures.
ROCM_ARCHITECTURES = ["gfx942", "gfx950", "gfx1100", "gfx1201"]


class ConvolutionStrategy(str, Enum):
    """ROCm convolution lowering strategy for TileAndFuse pipeline."""

    igemm = "igemm"
    direct = "direct"
    both = "both"


@dataclass
class ConvToIgemmInfo:
    """
    Stores information about convolution to IGEMM transformation.
    Used by get_padding_conv_sizes to calculate padding_conv attribute.

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
class LLVMGPUContractionKnobs(common.KnobAssignment):
    # Problem Size.
    M: int
    N: int
    K: int

    # Z3 numeric selections.
    tile_m: int
    tile_n: int
    tile_k: int
    wg_x: int
    wg_y: int
    wg_z: int
    subgroup_m_cnt: int
    subgroup_n_cnt: int
    intrinsic_mn: int
    intrinsic_k: int
    subgroup_m: int
    subgroup_n: int


@dataclass
class ConvolutionKnobs(common.KnobAssignment):
    pass


@dataclass
class AttentionKnobs(common.KnobAssignment):
    pass


def get_compatible_mma_intrinsics(
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic],
    allow_virtual_mma: bool = False,
) -> list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic]:
    def is_compatible(
        mma: iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic,
    ) -> bool:
        # Filter out virtual intrinsics unless explicitly allowed (for attention ops).
        is_virtual = isinstance(mma, iree_gpu.VirtualMMAIntrinsic)
        if is_virtual and not allow_virtual_mma:
            return False

        mma_attr = (
            iree_gpu.VirtualMMAAttr.get(mma)
            if is_virtual
            else iree_gpu.MMAAttr.get(mma)
        )
        a_type, b_type, c_type = mma_attr.abc_element_types
        return (
            lhs_type.element_type == a_type
            and rhs_type.element_type == b_type
            and res_type.element_type == c_type
        )

    return list(filter(is_compatible, mma_intrinsics))


# Generate a config dictionary used in translation_info attribute.
def get_translation_info_config(
    pipeline_options: iree_gpu.PipelineOptionsAttr, waves_per_eu: int
) -> ir.DictAttr:
    """
    Example IR
    translation_info = #iree_codegen.translation_info<
                    pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64,
                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
                     llvm_func_attrs = {"amdgpu-waves-per-eu" = "3"}
                    }
                >
    """
    waves_per_eu_str = str(waves_per_eu)

    # Create the waves_per_eu dictionary attribute.
    waves_per_eu_dict = ir.DictAttr.get(
        {WAVES_PER_EU_KEY: ir.StringAttr.get(waves_per_eu_str)}
    )

    config_dict = ir.DictAttr.get(
        {
            common.GPU_PIPELINE_OPTIONS_KEY: pipeline_options,
            common.LLVM_FUNC_ATTRS_KEY: waves_per_eu_dict,
        }
    )

    return config_dict


def get_attention_decomposition_config(
    tuner_ctx: common.TunerContext,
    qk_lowering_config: iree_gpu.LoweringConfigAttr,
    pv_lowering_config: iree_gpu.LoweringConfigAttr,
) -> ir.DictAttr:
    """
    Constructs the decomposition config for an attention op, embedding
    separate lowering configs for QK and PV matmuls.
    """

    ctx = tuner_ctx.mlir_ctx
    qk_attrs_dict = {
        "attention_qk_matmul": ir.UnitAttr.get(ctx),
        "lowering_config": qk_lowering_config,
    }
    qk_attr_dict = ir.DictAttr.get(qk_attrs_dict, context=ctx)

    pv_attrs_dict = {
        "attention_pv_matmul": ir.UnitAttr.get(ctx),
        "lowering_config": pv_lowering_config,
    }
    pv_attr_dict = ir.DictAttr.get(pv_attrs_dict, context=ctx)

    decomposition_config_dict = {
        "qk_attrs": qk_attr_dict,
        "pv_attrs": pv_attr_dict,
    }

    return ir.DictAttr.get(decomposition_config_dict, context=ctx)


# Implemented the logic from IREE side:
# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L382-L467.
def get_padding_conv_sizes(
    bounds: list[int],
    padding_sizes: list[int],
    igemm_loop_iterators: list[str],
    conv_to_igemm_info: ConvToIgemmInfo,
    convolution_dims: linalg.ConvolutionDimensions,
) -> Optional[list[int]]:
    """
    Computes padding_conv by mapping padding from IGEMM space to convolution space.

    Args:
        bounds: Loop bounds for each dimension.
        padding_sizes: Padding sizes in IGEMM dimension space (M, N, K).
        igemm_loop_iterators: IGEMM loop iterator type strings ('"reduction"' or '"parallel"').
        conv_to_igemm_info: Convolution to IGEMM transformation info.
        convolution_dims: Original convolution dimensions.

    Returns:
        Padding sizes in convolution dimension space, or None if no padding
        is needed along original convolution dimensions.
    """
    # Skip padding convolution for NCHW layout (spatial dimensions are last).
    if conv_to_igemm_info.is_spatial_dim_last:
        return None

    conv_to_igemm = conv_to_igemm_info.conv_to_igemm_dim
    padded_igemm_dims = set()
    input_channel_dims = set(convolution_dims.input_channel)

    padding_conv_sizes = [0] * len(conv_to_igemm)

    # For batch-last layout (e.g., CHWN), only pad the batch dimension to avoid
    # introducing pad op as the producer of collapse_shape op which may cause fusion problem.
    if conv_to_igemm_info.is_batch_dim_last:
        last_batch_dim = list(convolution_dims.batch)[-1]
        igemm_batch_pos = conv_to_igemm[last_batch_dim]

        if (
            padding_sizes[igemm_batch_pos]
            and bounds[igemm_batch_pos] % padding_sizes[igemm_batch_pos] == 0
        ):
            return None

        padding_conv_sizes[last_batch_dim] = padding_sizes[igemm_batch_pos]
        return padding_conv_sizes

    for conv_dim, igemm_pos in conv_to_igemm.items():
        if igemm_loop_iterators[igemm_pos] == '"reduction"':
            # Skip filter loop dimensions (reduction dims that aren't input channels).
            # Only pad input channel dims. If we need to pad filter dims, then we
            # would rather just do padding on the IGEMM instead.
            if conv_dim not in input_channel_dims:
                continue

            # Skip conv padding for input channel dims if already divisible by padding size.
            if (
                padding_sizes[igemm_pos]
                and bounds[igemm_pos] % padding_sizes[igemm_pos] == 0
            ):
                padded_igemm_dims.add(igemm_pos)
                continue

            # Multiple input channel dims for a single IGEMMPos is not supported.
            if igemm_pos in padded_igemm_dims:
                return None

            input_channel_size = conv_to_igemm_info.input_channel_dim_to_size.get(
                conv_dim, 0
            )
            is_input_channel_size_small = (
                padding_sizes[igemm_pos] // input_channel_size > 2
            )

            # If the input channel dimension is much smaller than the padding size,
            # skip padding along that dimension while still padding the others.
            if is_input_channel_size_small:
                padding_conv_sizes[conv_dim] = 0
            else:
                padding_conv_sizes[conv_dim] = padding_sizes[igemm_pos]

            padded_igemm_dims.add(igemm_pos)
            continue

        # Multiple padded parallel dims mapping to the same IGEMM dim is not supported.
        if padding_sizes[igemm_pos] and igemm_pos in padded_igemm_dims:
            return None

        padding_conv_sizes[conv_dim] = padding_sizes[igemm_pos]
        padded_igemm_dims.add(igemm_pos)

    # Ensure that all dimensions have been padded.
    if len(padded_igemm_dims) != len(padding_sizes):
        return None

    return padding_conv_sizes


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
    """Result from rocprof benchmarking (converted to BenchmarkResult by libtuner)."""

    candidate_id: int
    time: float
    device_id: str


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
