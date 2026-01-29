# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest constraint_generator_test.py
"""

import z3  # type: ignore

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func, linalg  # type: ignore

from amdsharktuner import (
    common,
    constraint_generator,
)


def build_func_with_matmul(
    module: ir.Module,
    m: int,
    n: int,
    k: int,
    lhs_type: ir.Type,
    rhs_type: ir.Type,
    res_type: ir.Type,
) -> None:
    a_type = ir.RankedTensorType.get((m, k), lhs_type)
    b_type = ir.RankedTensorType.get((k, n), rhs_type)
    c_type = ir.RankedTensorType.get((m, n), res_type)

    dim_m = ir.AffineDimExpr.get(0)
    dim_n = ir.AffineDimExpr.get(1)
    dim_k = ir.AffineDimExpr.get(2)
    a_map = ir.AffineMap.get(3, 0, [dim_m, dim_k])
    b_map = ir.AffineMap.get(3, 0, [dim_k, dim_n])
    c_map = ir.AffineMap.get(3, 0, [dim_m, dim_n])

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(a_type, b_type, c_type)
        def named_matmul(a: ir.Value, b: ir.Value, c: ir.Value) -> None:
            matmul_op = linalg.MatmulOp(
                result_tensors=[c_type],
                inputs=[a, b],
                outputs=[c],
                indexing_maps=[a_map, b_map, c_map],
            )
            matmul_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def build_func_with_conv2d_nhwc_hwcf(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    kernel_type: ir.Type,
    output_type: ir.Type,
) -> None:
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    kernel_tensor_type = ir.RankedTensorType.get(kernel_shape, kernel_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, kernel_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            conv_op = linalg.Conv2DNhwcHwcfOp(
                inputs=[arg0, arg1],
                outputs=[arg2],
                result_tensors=[output_tensor_type],
            )
            conv_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def test_ContractionZ3Constants_to_meta() -> None:
    matmul_size = common.ContractionSizes(M=[1, 2], N=[1], K=[1], B=[1])
    orig = constraint_generator.ContractionZ3Constants.from_sizes(matmul_size)
    meta = orig.to_meta()
    expected = {
        "m_vals": ["m0", "m1"],
        "n_vals": ["n0"],
        "k_vals": ["k0"],
        "subgroup_m_vals": ["subgroup_m0", "subgroup_m1"],
        "subgroup_n_vals": ["subgroup_n0"],
        "subgroup_size": "subgroup_size",
        "intrinsic_mn": "intrinsic_mn",
        "intrinsic_k": "intrinsic_k",
        "wg_x": "wg_x",
        "wg_y": "wg_y",
        "wg_z": "wg_z",
        "sg_m_cnt": "sg_m_cnt",
        "sg_n_cnt": "sg_n_cnt",
    }
    assert meta == expected


def test_ContractionZ3Constants_from_meta_dict() -> None:
    meta: dict[str, str | list[str]] = {
        "m_vals": ["m0", "m1"],
        "n_vals": ["n0"],
        "k_vals": ["k0"],
        "subgroup_m_vals": ["subgroup_m0", "subgroup_m1"],
        "subgroup_n_vals": ["subgroup_n0"],
        "subgroup_size": "subgroup_size",
        "intrinsic_mn": "intrinsic_mn",
        "intrinsic_k": "intrinsic_k",
        "wg_x": "wg_x",
        "wg_y": "wg_y",
        "wg_z": "wg_z",
        "sg_m_cnt": "sg_m_cnt",
        "sg_n_cnt": "sg_n_cnt",
    }

    ctx = z3.Context()
    recon = constraint_generator.ContractionZ3Constants.from_meta(meta=meta, ctx=ctx)

    assert recon.m_vals == [z3.Int(f"m{i}", ctx) for i in range(2)]
    assert recon.n_vals == [z3.Int(f"n0", ctx)]
    assert recon.k_vals == [z3.Int(f"k0", ctx)]
    assert recon.subgroup_m_vals == [z3.Int(f"subgroup_m{i}", ctx) for i in range(2)]
    assert recon.subgroup_n_vals == [z3.Int(f"subgroup_n0", ctx)]

    assert recon.subgroup_size == z3.Int("subgroup_size", ctx)
    assert recon.intrinsic_mn == z3.Int("intrinsic_mn", ctx)
    assert recon.intrinsic_k == z3.Int("intrinsic_k", ctx)
    assert recon.wg_x == z3.Int("wg_x", ctx)
    assert recon.wg_y == z3.Int("wg_y", ctx)
    assert recon.wg_z == z3.Int("wg_z", ctx)
    assert recon.sg_m_cnt == z3.Int("sg_m_cnt", ctx)
    assert recon.sg_n_cnt == z3.Int("sg_n_cnt", ctx)


def test_Z3Constants_meta_roundtrip() -> None:
    matmul_size = common.ContractionSizes(M=[1], N=[1], K=[1], B=[1])
    orig = constraint_generator.ContractionZ3Constants.from_sizes(matmul_size)
    meta = orig.to_meta()

    ctx = z3.Context()
    recon = constraint_generator.ContractionZ3Constants.from_meta(meta=meta, ctx=ctx)

    for f in constraint_generator.fields(orig):
        orig_field = getattr(orig, f.name)
        recon_field = getattr(recon, f.name)

        if isinstance(orig_field, list):
            assert [v.decl().name() for v in orig_field] == [
                v.decl().name() for v in recon_field
            ]
        else:
            assert orig_field.decl().name() == recon_field.decl().name()


def test_get_z3_solutions() -> None:
    matmul_size = common.ContractionSizes(M=[1], N=[1], K=[1], B=[1])
    z3_constants = constraint_generator.ContractionZ3Constants.from_sizes(matmul_size)

    solver = z3.Solver()
    for v in z3_constants.symbols:
        if v is not z3_constants.wg_x:
            solver.add(v == 0)
    solver.add(z3.Or(z3_constants.wg_x == 0, z3_constants.wg_x == 1))

    z3_constraint_set = constraint_generator.ConstraintSet(
        solver=solver, z3_constants=z3_constants
    )

    results = list(constraint_generator.get_z3_solutions(z3_constraint_set))
    pairs = {(res.wg_x, res.wg_y) for res in results}

    assert pairs == {(0, 0), (1, 0)}
    assert len(results) == 2

    solver.add(z3_constants.wg_y == 1)
    z3_constraint_set = constraint_generator.ConstraintSet(
        solver=solver, z3_constants=z3_constants
    )
    results = list(constraint_generator.get_z3_solutions(z3_constraint_set))
    assert results == []
