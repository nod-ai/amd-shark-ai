# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest constraint_generator_test.py
"""

from amdsharktuner import common, constraint_generator


def test_get_z3_solutions() -> None:
    matmul_size = common.ContractionSizes(M=[1], N=[1], K=[1], B=[1])
    z3_constants = constraint_generator.ContractionZ3Constants.from_sizes(matmul_size)

    solver = constraint_generator.z3.Solver()
    for v in z3_constants.symbols:
        if v is not z3_constants.wg_x:
            solver.add(v == 0)
    solver.add(
        constraint_generator.z3.Or(z3_constants.wg_x == 0, z3_constants.wg_x == 1)
    )

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
