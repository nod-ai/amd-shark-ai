# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Generator

import pytest
import z3  # type: ignore

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore

from amdsharktuner import candidate_gen, common


@pytest.fixture
def sample_constraints_op() -> Generator[iree_codegen.ConstraintsOp, None, None]:
    test_mlir_str = """
        module {
            iree_codegen.smt.constraints
                target = <set = 0>,
                pipeline = #iree_gpu.pipeline<VectorDistribute>,
                knobs = {test = #iree_codegen.smt.int_knob<"test">,
                    workgroup = [#iree_codegen.smt.int_knob<"wg_m">,
                                    #iree_codegen.smt.int_knob<"wg_n">,
                                    #iree_codegen.smt.int_knob<"wg_k">],
                    workgroup_size = [#iree_codegen.smt.int_knob<"wg_x">,
                                         #iree_codegen.smt.int_knob<"wg_y">,
                                         #iree_codegen.smt.int_knob<"wg_z">],
                    subgroup_size = #iree_codegen.smt.int_knob<"sg_size">,
                    mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx",
                            ["opt_a", "opt_b", "opt_c"]>,
                    subgroup_basis = [[#iree_codegen.smt.int_knob<"sg_x">,
                                        #iree_codegen.smt.int_knob<"sg_y">],
                                      [0, 1]]}
                dims() {
                }
        }
    """
    with ir.Context():
        module = ir.Module.parse(test_mlir_str)
        ops = ir.get_ops_of_type(module, iree_codegen.ConstraintsOp)
        yield ops[0]


@pytest.fixture
def sample_knob_assignment() -> dict[str, int]:
    assignment = {
        "test": 100,
        "wg_m": 128,
        "wg_n": 64,
        "wg_k": 64,
        "wg_x": 64,
        "wg_y": 2,
        "wg_z": 1,
        "sg_size": 64,
        "mma_idx": 1,
        "sg_x": 2,
        "sg_y": 4,
    }
    return assignment


def test_get_z3_assignment_from_model() -> None:
    a = z3.Int("a")
    b = z3.Int("b")
    solver = z3.Solver()
    solver.add(a == 4)
    solver.add(b == 6)
    assert solver.check() == z3.sat
    model = solver.model()
    symbols = common.SMTKnobSymbols({"a": a, "b": b, "sum": a + b})
    result = candidate_gen.get_z3_assignment_from_model(model, symbols)
    assert result["a"] == 4
    assert result["b"] == 6
    assert result["sum"] == 10


def test_get_knobs_from_constraint_op(
    sample_constraints_op: iree_codegen.ConstraintsOp,
    sample_knob_assignment: dict[str, int],
) -> None:
    symbols = candidate_gen.get_knobs_from_constraint_op(
        sample_constraints_op, z3_ctx=z3.Context()
    )
    expected_keys = sample_knob_assignment.keys()

    assert set(symbols.keys()) == expected_keys
    for name, expr in symbols.items():
        assert z3.is_int(expr)
        # Check Z3 variable is created with the expected name.
        assert expr.decl().name() == name


def test_get_knobs_from_constraint_op_ignores_constant_entries() -> None:
    mlir = """
    module {
        iree_codegen.smt.constraints
            target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {
                workgroup = [#iree_codegen.smt.int_knob<"wg_m">, 1],
                subgroup_basis = [[#iree_codegen.smt.int_knob<"sg_m">, 1], [0, 1]],
                subgroup_size = #iree_codegen.smt.int_knob<"sg_size">,
                literal = 42
            }
            dims() {
            }
    }
    """
    with ir.Context():
        module = ir.Module.parse(mlir)
        ops = ir.get_ops_of_type(module, iree_codegen.ConstraintsOp)
        symbols = candidate_gen.get_knobs_from_constraint_op(
            ops[0], z3_ctx=z3.Context()
        )

    assert set(symbols.keys()) == {"wg_m", "sg_m", "sg_size"}


def test_generate_solutions_yields_assignments() -> None:
    unsolvable_mlir_str = """
    module {
        iree_codegen.smt.constraints
            target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {wg_m = #iree_codegen.smt.int_knob<"wg_m">}
            dims() {
            ^bb0:
            %v = iree_codegen.smt.knob "wg_m" : !smt.int
            %c4 = smt.int.constant 4
            %c8 = smt.int.constant 8
            %ge = smt.int.cmp ge %v, %c8
            %le = smt.int.cmp le %v, %c4
            iree_codegen.smt.assert %ge, "wg_m >= 8" : !smt.bool
            iree_codegen.smt.assert %le, "wg_m <= 4" : !smt.bool
            }
    }
    """
    with ir.Context():
        module = ir.Module.parse(unsolvable_mlir_str)
        ops = ir.get_ops_of_type(module, iree_codegen.ConstraintsOp)
        solutions = list(
            candidate_gen.generate_solutions_from_constraint_op(
                ops[0], z3_ctx=z3.Context()
            )
        )
    assert (
        len(solutions) == 0
    ), f"Expected no solutions for unsolvable constraints, got {len(solutions)} solutions."

    solvable_mlir_str = """
    module {
        iree_codegen.smt.constraints
            target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {wg_m = #iree_codegen.smt.int_knob<"wg_m">}
            dims() {
            ^bb0:
            %v = iree_codegen.smt.knob "wg_m" : !smt.int
            %c4 = smt.int.constant 4
            %c8 = smt.int.constant 8
            %ge = smt.int.cmp ge %v, %c4
            %le = smt.int.cmp le %v, %c8
            iree_codegen.smt.assert %ge, "wg_m >= 4" : !smt.bool
            iree_codegen.smt.assert %le, "wg_m <= 8" : !smt.bool
            }
    }
    """
    with ir.Context():
        module = ir.Module.parse(solvable_mlir_str)
        ops = ir.get_ops_of_type(module, iree_codegen.ConstraintsOp)
        solutions = list(
            candidate_gen.generate_solutions_from_constraint_op(
                ops[0], z3_ctx=z3.Context()
            )
        )
    assert len(solutions) > 0, "Expected solutions for solvable constraints."
    seen: set[tuple] = set()
    for sol in solutions:
        key = tuple(sorted(sol.items()))
        assert key not in seen, f"Duplicate solution: {sol}"
        seen.add(key)
        assert isinstance(sol, dict)
        assert "wg_m" in sol
        assert 4 <= sol["wg_m"] <= 8


def test_generate_constraint_solutions_from_constraint_op() -> None:
    mlir = """
    module {
        iree_codegen.smt.constraints
            target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {wg_m = #iree_codegen.smt.int_knob<"wg_m">}
            dims() {
            ^bb0:
            %v = iree_codegen.smt.knob "wg_m" : !smt.int
            %c4 = smt.int.constant 4
            %c8 = smt.int.constant 8
            %ge = smt.int.cmp ge %v, %c4
            %le = smt.int.cmp le %v, %c8
            iree_codegen.smt.assert %ge, "wg_m >= 4" : !smt.bool
            iree_codegen.smt.assert %le, "wg_m <= 8" : !smt.bool
            }
    }
    """
    with ir.Context() as ctx:
        module = ir.Module.parse(mlir, ctx)
        solutions = list(candidate_gen.generate_solutions(module, ctx))
    assert len(solutions) > 0
    for solution in solutions:
        assert isinstance(solution, candidate_gen.ConstraintSolution)
        assert isinstance(solution.constraints_op, iree_codegen.ConstraintsOp)
        assert isinstance(solution.knob_assignments, dict)
