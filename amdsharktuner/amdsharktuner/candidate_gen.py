# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator

import iree.compiler as ireec  # type: ignore
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore
import z3  # type: ignore

from . import (
    common,
    process_utils,
    spec_builder,
)
from .rocm import rocm_common, rocm_tuners
from .tuner_base import DispatchTuner

tune_logger = logging.getLogger("tune")


@dataclass(frozen=True)
class ConstraintSolution:
    constraints_module: ir.Module
    constraints_op: iree_codegen.ConstraintsOp
    knob_assignments: common.SMTKnobAssignments


def get_z3_assignment_from_model(
    model: z3.ModelRef,
    z3_const_exprs: common.SMTKnobSymbols,
) -> common.SMTKnobAssignments:
    def get_z3_const_val(v: z3.ExprRef) -> int:
        val = model.eval(v)
        assert z3.is_int_value(
            val
        ), f"Unassigned or non-concrete constant: {v} -> {val}"
        return val.as_long()

    return common.SMTKnobAssignments(
        {name: get_z3_const_val(expr) for name, expr in z3_const_exprs.items()}
    )


def get_knobs_from_constraint_op(
    constraints_op: iree_codegen.ConstraintsOp,
    z3_ctx: z3.Context,
) -> common.SMTKnobSymbols:
    knob_names: list[str] = []

    def collect(attr: ir.Attribute) -> None:
        match attr:
            case iree_codegen.IntKnobAttr() | iree_codegen.OneOfKnobAttr():
                knob_names.append(attr.name)
            case ir.IntegerAttr():
                return
            case ir.ArrayAttr():
                for elem in attr:
                    collect(elem)
            case ir.DictAttr():
                for entry in attr:
                    collect(entry.attr)
            case _:
                raise TypeError(f"Unknown knob attribute type: {type(attr)}")

    collect(constraints_op.knobs)

    return common.SMTKnobSymbols(
        {name: z3.Int(name, ctx=z3_ctx) for name in knob_names}
    )


def generate_solutions_from_constraint_op(
    constraints_op: iree_codegen.ConstraintsOp,
    z3_ctx: z3.Context,
) -> Iterator[common.SMTKnobAssignments]:
    with constraints_op.operation.context:
        smtlib = iree_codegen.convert_constraints_op_to_smtlib(
            constraints_op, emit_reset=False
        )
    if "(reset)" in smtlib:
        raise RuntimeError(f"Unexpected reset string in SMTLIB: \n{smtlib}")

    z3_const_exprs = get_knobs_from_constraint_op(constraints_op, z3_ctx)
    z3_vars = list(z3_const_exprs.values())

    solver = z3.Solver(ctx=z3_ctx)
    solver.add(z3.parse_smt2_string(smtlib, ctx=z3_ctx))

    count = 0
    while solver.check() == z3.sat:
        model = solver.model()
        solver.add(z3.Or([v != model.eval(v, model_completion=True) for v in z3_vars]))

        z3_assignment = get_z3_assignment_from_model(model, z3_const_exprs)
        count += 1
        tune_logger.debug(f"Solution #{count}: {z3_assignment}")
        yield z3_assignment


def get_supported_dispatch_tuners(
    target_arch: str,
    codegen_pipeline: iree_gpu.LoweringPipeline,
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
            "--iree-codegen-add-tuner-attributes during compilation?"
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
    input_module: ir.Module,
    mlir_ctx: ir.Context,
) -> Iterator[ConstraintSolution]:
    constraints_module = get_constraints_module(input_module, mlir_ctx)
    constraints_ops = ir.get_ops_of_type(constraints_module, iree_codegen.ConstraintsOp)
    tune_logger.debug(f"Found {len(constraints_ops)} constraints ops")
    if len(constraints_ops) == 0:
        raise RuntimeError("Expected at least one iree_codegen.smt.constraints op")
    # TODO(Amily): Tuner currently supports only one ConstraintsOp.
    if len(constraints_ops) > 1:
        tune_logger.warning(
            f"Found {len(constraints_ops)} iree_codegen.smt.constraints ops. "
            "Using the first one because tuner currently supports one ConstraintsOp."
        )
    constraints_op = constraints_ops[0]
    z3_ctx = z3.Context()
    for knob_assignments in generate_solutions_from_constraint_op(
        constraints_op, z3_ctx=z3_ctx
    ):
        yield ConstraintSolution(
            constraints_module=constraints_module,
            constraints_op=constraints_op,
            knob_assignments=knob_assignments,
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
    iree_opt: str = ireec.binaries.find_tool("iree-opt")  # type: ignore[attr-defined]
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


def get_constraints_module(input_module: ir.Module, mlir_ctx: ir.Context) -> ir.Module:
    """Run `iree-compile` and return IR dump stderr as an in-memory module."""
    iree_compile: str = ireec.binaries.find_tool("iree-compile")  # type: ignore[attr-defined]
    module_str = str(input_module)
    command = [
        iree_compile,
        "-",
        "--iree-codegen-experimental-verify-pipeline-constraints",
        "--mlir-print-ir-after=iree-codegen-insert-smt-constraints",
    ]
    tune_logger.debug("Running iree-compile to insert SMT constraints.")
    result = process_utils.run_command(
        process_utils.RunPack(
            command=command,
            check=False,
            stdin=module_str,
            capture_stdout=False,
        )
    )
    try:
        with mlir_ctx:
            constraints_module = ir.Module.parse(result.process_res.stderr)
    except Exception as e:
        error_msg = f"Failed to insert SMT constraints into input module: {e}"
        tune_logger.error(error_msg)
        raise RuntimeError(error_msg)
    return constraints_module
