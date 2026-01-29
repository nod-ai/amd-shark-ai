# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import logging
from pathlib import Path
from typing import Optional, Iterator

import iree.compiler as ireec  # type: ignore
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from . import (
    common,
    process_utils,
    spec_builder,
)
from .rocm import rocm_common, rocm_dispatch_constraints, rocm_tuners
from .tuner_base import DispatchTuner

tune_logger = logging.getLogger("tune")


def get_supported_dispatch_tuners(
    target_arch: str,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
) -> list[type[DispatchTuner]]:
    """Get supported dispatch tuners for the given target architecture and pipeline."""
    # TODO(Bangtian): Use `target.getBackend() == "rocm"` once backend name is exposed
    # in TargetInfo. Currently using "gfx" prefix matching as a workaround.
    is_rocm_arch = target_arch.startswith("gfx")
    if not is_rocm_arch:
        tune_logger.warning(
            f"Target architecture '{target_arch}' is not a ROCm architecture. "
            f"Only ROCm (gfx*) architectures are currently supported."
        )
        return []

    # Allow tuning on untested architectures with a warning, since the tuning
    # logic may still work even if we haven't validated it.
    if target_arch not in rocm_common.ROCM_ARCHITECTURES:
        tune_logger.warning(
            f"Target architecture '{target_arch}' is not tested. "
            f"Tested ROCm architectures: {rocm_common.ROCM_ARCHITECTURES}. "
            f"Proceeding with tuning anyway."
        )

    # Get tuners for ROCm backend.
    return rocm_tuners.get_tuners_for_pipeline(codegen_pipeline)


def instantiate_dispatch_tuner(
    input_module: ir.Module,
    tuner_ctx: common.TunerContext,
    dispatch_tuners: list[type[DispatchTuner]],
) -> Optional[DispatchTuner]:
    """Find and instantiate a suitable dispatch tuner for the input module."""
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    if len(root_op_list) == 0:
        tune_logger.error(
            "No root ops found. Did you forget to pass "
            "--iree-config-add-tuner-attributes during compilation?"
        )
        return None
    elif len(root_op_list) > 1:
        tune_logger.error("Multiple root ops found. Only one is currently supported.")
        return None

    root_op = root_op_list[0]

    dispatch_tuner: Optional[DispatchTuner] = None
    for tuner_class in dispatch_tuners:
        if tuner_class.supports_root_op(root_op):
            dispatch_tuner = tuner_class(root_op, tuner_ctx)
            break

    if not dispatch_tuner:
        tune_logger.error(
            "No suitable dispatch tuner found for the root operation. "
            "The operation may not be supported by the tuner yet."
        )

    return dispatch_tuner


def generate_solutions(
    dispatch_tuner: DispatchTuner,
    target_info: iree_gpu.TargetInfo,
    tuner_context: common.TunerContext,
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints.
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: rocm_dispatch_constraints.PipelineOptionsSearchSpace = rocm_dispatch_constraints.PipelineOptionsSearchSpace(),
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
) -> Iterator[list[common.TuningConfiguration]]:
    if target_info.arch not in rocm_common.ROCM_ARCHITECTURES:
        print(f"Warning: Untested architecture '{target_info.arch}'.")

    constraint_generator = dispatch_tuner.get_constraint_generator()

    return constraint_generator.generate_solutions(
        tuner_context,
        target_info,
        num_subgroups=num_subgroups,
        allowed_waves_per_eu=allowed_waves_per_eu,
        pipeline_options_search_space=pipeline_options_search_space,
    )


def generate_configs_and_td_specs(
    dispatch_tuner: DispatchTuner,
    input_module: ir.Module,  # In-memory module to be tuned.
    solutions: list[list[common.TuningConfiguration]],
) -> list[ir.Module]:
    # Index 0 is reserved for default config, so it gets a placeholder spec.
    config_specs: list[ir.Module] = [
        spec_builder.get_placeholder_spec(input_module.context)
    ]

    for i, config in enumerate(solutions):
        tune_logger.debug(f"Solution #{i+1}: {config}")
        td_spec_module = dispatch_tuner.get_td_spec(config)
        assert td_spec_module, "Failed to generate transform dialect spec"
        config_specs.append(td_spec_module)

    tune_logger.debug(f"Generated {len(config_specs)} tuning specs")

    return config_specs


# The `strip_root_op_attr` and `strip_compilation_info` functions are used for
# getting consistent inputs to the compilation step in tuning. Inputs may come
# in with lowering configs, translation info, and root_op attrs when the input
# is a benchmark, but not when the input is a source MLIR file. Stripping the
# info makes the inputs to compilation consistent, and allows for overwriting
# the compilation info with generated TD specs during codegen.
def strip_root_op_attr(module: ir.Module):
    root_ops: list[ir.Operation] = iree_codegen.get_tuner_root_ops(module)
    for root_op in root_ops:
        assert (
            spec_builder.ROOT_OP_ATTR_NAME in root_op.opview.attributes
        ), f"expected root op to have '{spec_builder.ROOT_OP_ATTR_NAME}' attr"
        del root_op.opview.attributes[spec_builder.ROOT_OP_ATTR_NAME]


# See the above comment for `strip_root_op_attr`.
def strip_compilation_info(input_path: Path) -> str:
    # Strip compilation info from the source and save the stripped IR.
    iree_opt = ireec.binaries.find_tool("iree-opt")  # type: ignore
    assert iree_opt, "iree-opt tool not found"
    strip_command = [
        iree_opt,
        f"{input_path}",
        f"--iree-codegen-strip-compilation-info",
    ]
    result = process_utils.run_command(
        process_utils.RunPack(
            command=strip_command,
            check=True,
        )
    )
    assert (
        result.process_res is not None
    ), "expected result from stripping compilation info"
    return result.process_res.stdout
