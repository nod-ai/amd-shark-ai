# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Generator

import pytest
import z3  # type: ignore

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from amdsharktuner import common, smt_candidate_gen


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
                    subgroup_basis = {
                            counts = [#iree_codegen.smt.int_knob<"sg_x">,
                                        #iree_codegen.smt.int_knob<"sg_y">],
                            mapping = [#iree_codegen.smt.int_knob<"map_0">,
                                        #iree_codegen.smt.int_knob<"map_1">]}}
                dims() {
                }
        }
    """
    with ir.Context():
        module = ir.Module.parse(test_mlir_str)
        ops = ir.get_ops_of_type(module, iree_codegen.ConstraintsOp)
        yield ops[0]


@pytest.fixture
def sample_knob_assignment() -> common.SMTKnobAssignment:
    assignment = common.SMTKnobAssignment(
        {
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
            "map_0": 0,
            "map_1": 1,
        }
    )
    return assignment


def test_get_z3_assignment_from_model() -> None:
    a = z3.Int("a")
    b = z3.Int("b")
    solver = z3.Solver()
    solver.add(a == 4)
    solver.add(b == 6)
    assert solver.check() == z3.sat
    model = solver.model()
    symbols = common.KnobSymbols({"a": a, "b": b, "sum": a + b})
    result = smt_candidate_gen.get_z3_assignment_from_model(model, symbols)
    assert result["a"] == 4
    assert result["b"] == 6
    assert result["sum"] == 10


def test_resolve_knob_array_attr_template() -> None:
    with ir.Context():
        arr = ir.Attribute.parse(
            '[#iree_codegen.smt.int_knob<"wg_m">, '
            '#iree_codegen.smt.int_knob<"wg_n">]'
        )
        assignment = common.SMTKnobAssignment({"wg_m": 64, "wg_n": 128})
        result = smt_candidate_gen._resolve_knob_array_attr_template(arr, assignment)
        assert result == [64, 128]

        with pytest.raises(AssertionError, match="wg_m"):
            smt_candidate_gen._resolve_knob_array_attr_template(
                arr, common.SMTKnobAssignment({})
            )


def test_get_template_entry() -> None:
    with ir.Context():
        knob_template = ir.Attribute.parse(
            '{wg_m = #iree_codegen.smt.int_knob<"wg_m">}'
        )
    assert isinstance(knob_template, ir.DictAttr)
    key: common.AttrKey = common.AttrKey("wg_m", iree_codegen.IntKnobAttr)
    result = smt_candidate_gen._get_template_entry(knob_template, key)

    assert result is not None
    assert isinstance(result, iree_codegen.IntKnobAttr)
    assert result.name == "wg_m"
    key = common.AttrKey("test", iree_codegen.IntKnobAttr)
    result = smt_candidate_gen._get_template_entry(knob_template, key)
    assert result is None


def test_get_knobs_from_constraint_op(
    sample_constraints_op: iree_codegen.ConstraintsOp,
    sample_knob_assignment: common.SMTKnobAssignment,
) -> None:
    symbols = smt_candidate_gen.get_knobs_from_constraint_op(
        sample_constraints_op, z3_ctx=z3.Context()
    )
    expected_keys = sample_knob_assignment.keys()

    assert set(symbols.keys()) == expected_keys
    for name, expr in symbols.items():
        assert z3.is_int(expr)
        # Check Z3 variable is created with the expected name.
        assert expr.decl().name() == name


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
            smt_candidate_gen.generate_solutions_from_constraint_op(
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
            smt_candidate_gen.generate_solutions_from_constraint_op(
                ops[0], z3_ctx=z3.Context()
            )
        )
    assert len(solutions) > 0, "Expected solutions for solvable constraints."
    seen: set[tuple] = set()
    for sol in solutions:
        key = tuple(sorted(sol.items()))
        assert key not in seen, f"Duplicate solution: {sol}"
        seen.add(key)
        assert isinstance(sol, common.SMTKnobAssignment)
        assert "wg_m" in sol
        assert 4 <= sol["wg_m"] <= 8


def test_build_lowering_config_attr(
    sample_constraints_op: iree_codegen.ConstraintsOp,
    sample_knob_assignment: common.SMTKnobAssignment,
) -> None:
    config = smt_candidate_gen.GPUCompilationInfoBuilder.LoweringConfig.build_lowering_config_attr(
        sample_constraints_op, sample_knob_assignment
    )
    assert isinstance(config, iree_gpu.LoweringConfigAttr)
    config_dict = config.attributes

    assert "test" not in config_dict
    assert "workgroup" in config_dict
    assert str(config_dict["workgroup"]) == "[128, 64, 64]"
    assert str(config_dict["mma_kind"]) == '"opt_b"'
    assert str(config_dict["subgroup_basis"]) == "[[2, 4], [0, 1]]"
    assert "workgroup_size" not in config_dict
    assert "subgroup_size" not in config_dict


def test_build_translation_info_attr(
    sample_constraints_op: iree_codegen.ConstraintsOp,
    sample_knob_assignment: common.SMTKnobAssignment,
) -> None:
    translation_info = smt_candidate_gen.GPUCompilationInfoBuilder.TranslationInfo.build_translation_info_attr(
        sample_constraints_op, sample_knob_assignment
    )
    assert isinstance(translation_info, iree_codegen.TranslationInfoAttr)

    assert "test" not in str(translation_info)
    assert str(translation_info.workgroup_size) == "[64, 2, 1]"
    assert translation_info.subgroup_size == 64


def test_build_compilation_info_attr(
    sample_constraints_op: iree_codegen.ConstraintsOp,
    sample_knob_assignment: common.SMTKnobAssignment,
) -> None:
    compilation_info = (
        smt_candidate_gen.GPUCompilationInfoBuilder.build_compilation_info_attr(
            sample_constraints_op, sample_knob_assignment
        )
    )
    assert isinstance(compilation_info, iree_codegen.CompilationInfoAttr)
    assert isinstance(compilation_info.lowering_config, iree_gpu.LoweringConfigAttr)
    assert isinstance(
        compilation_info.translation_info, iree_codegen.TranslationInfoAttr
    )
