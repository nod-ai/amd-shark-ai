# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import z3  # type: ignore
from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar
from dataclasses import dataclass, fields

from iree.compiler.dialects import iree_gpu  # type: ignore

from . import common


TScalar = TypeVar("TScalar", int, z3.ExprRef)


@dataclass(slots=True)
class ContractionConstantsBase(Generic[TScalar]):
    m_vals: list[TScalar]
    n_vals: list[TScalar]
    k_vals: list[TScalar]
    subgroup_m_vals: list[TScalar]
    subgroup_n_vals: list[TScalar]

    subgroup_size: TScalar
    intrinsic_mn: TScalar
    intrinsic_k: TScalar
    wg_x: TScalar
    wg_y: TScalar
    wg_z: TScalar
    sg_m_cnt: TScalar
    sg_n_cnt: TScalar


@dataclass(slots=True)
class ContractionZ3Assignment(ContractionConstantsBase[int]):
    """Interpretations of Z3 constants from a satisfying model."""

    pass


@dataclass(slots=True)
class ContractionZ3Constants(ContractionConstantsBase[z3.ExprRef]):
    """Z3 uninterpreted constants used in constraints."""

    @classmethod
    def from_sizes(
        cls, matmul_size: common.ContractionSizes
    ) -> "ContractionZ3Constants":
        M, N, K = matmul_size.M, matmul_size.N, matmul_size.K

        m_vals = [z3.Int(f"m{i}") for i in range(len(M))]
        n_vals = [z3.Int(f"n{i}") for i in range(len(N))]
        k_vals = [z3.Int(f"k{i}") for i in range(len(K))]
        subgroup_m_vals = [z3.Int(f"subgroup_m{i}") for i in range(len(M))]
        subgroup_n_vals = [z3.Int(f"subgroup_n{i}") for i in range(len(N))]

        subgroup_size = z3.Int("subgroup_size")
        intrinsic_mn = z3.Int("intrinsic_mn")
        intrinsic_k = z3.Int("intrinsic_k")
        wg_x, wg_y, wg_z = z3.Int("wg_x"), z3.Int("wg_y"), z3.Int("wg_z")
        sg_m_cnt = z3.Int("sg_m_cnt")
        sg_n_cnt = z3.Int("sg_n_cnt")

        return cls(
            m_vals=m_vals,
            n_vals=n_vals,
            k_vals=k_vals,
            subgroup_m_vals=subgroup_m_vals,
            subgroup_n_vals=subgroup_n_vals,
            subgroup_size=subgroup_size,
            intrinsic_mn=intrinsic_mn,
            intrinsic_k=intrinsic_k,
            wg_x=wg_x,
            wg_y=wg_y,
            wg_z=wg_z,
            sg_m_cnt=sg_m_cnt,
            sg_n_cnt=sg_n_cnt,
        )

    def to_meta(self) -> dict[str, str | list[str]]:
        """
        Serialize constants to their symbol names.

        Example:
        meta = {
            "m_vals": ["m0", "m1"],
            "subgroup_n_vals": ["subgroup_n0"],
            "intrinsic_mn": "intrinsic_mn",
            ...
        }
        """
        meta: dict[str, str | list[str]] = {}

        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, list):
                meta[f.name] = [v.decl().name() for v in attr]
            else:
                meta[f.name] = attr.decl().name()

        return meta

    @classmethod
    def from_meta(
        cls, meta: dict[str, str | list[str]], ctx: z3.Context
    ) -> "ContractionZ3Constants":
        """
        Reconstruct constants from serialized metadata.

        Z3 expressions are context-bound and cannot be shared across
        contexts. Providing `ctx` ensures that all reconstructed symbols
        belong to the same context and can be safely combined in constraints
        and solvers.

        Example:
        meta = {
            "m_vals": ["m0", "m1"],
            "subgroup_n_vals": ["subgroup_n0"],
            "intrinsic_mn": "intrinsic_mn",
            ...
        }
        kwargs = {
            "m_vals": [z3.Int("m0", ctx), z3.Int("m1", ctx)],
            "subgroup_n_vals": [z3.Int("subgroup_n0", ctx)],
            "intrinsic_mn": z3.Int("intrinsic_mn", ctx),
            ...
        }
        """
        kwargs = {}
        for f in fields(cls):
            value = meta[f.name]
            if isinstance(value, list):
                kwargs[f.name] = [z3.Int(name, ctx=ctx) for name in value]
            else:
                kwargs[f.name] = z3.Int(value, ctx=ctx)

        return cls(**kwargs)

    @property
    def symbols(self) -> list[z3.ExprRef]:
        """All constants whose values are extracted from the model."""
        vars_list: list[z3.ExprRef] = []
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, list):
                vars_list.extend(attr)
            else:
                vars_list.append(attr)
        return vars_list

    def extract(self, model: z3.ModelRef) -> ContractionZ3Assignment:
        """Extract a satisfying assignment from the model."""

        def get(v: z3.ExprRef) -> int:
            # Evaluate arbitrary expressions over a model, convert z3 expr to int.
            val = model.eval(v)
            assert z3.is_int_value(
                val
            ), f"Unassigned or non-concrete constant: {v} -> {val}"
            return val.as_long()

        return ContractionZ3Assignment(
            m_vals=[get(v) for v in self.m_vals],
            n_vals=[get(v) for v in self.n_vals],
            k_vals=[get(v) for v in self.k_vals],
            subgroup_m_vals=[get(v) for v in self.subgroup_m_vals],
            subgroup_n_vals=[get(v) for v in self.subgroup_n_vals],
            subgroup_size=get(self.subgroup_size),
            intrinsic_mn=get(self.intrinsic_mn),
            intrinsic_k=get(self.intrinsic_k),
            wg_x=get(self.wg_x),
            wg_y=get(self.wg_y),
            wg_z=get(self.wg_z),
            sg_m_cnt=get(self.sg_m_cnt),
            sg_n_cnt=get(self.sg_n_cnt),
        )


@dataclass
class ConstraintSet:
    """
    A container for a Z3 solver and the symbolic constants used in its constraints.

    `solver` is expected to be populated with all assertions defining a
    single contraction problem. No additional base constraints should be added
    after construction. The solver should be ready for Z3 `check()` and model enumeration.

    'z3_constants` contains the complete set of Z3 symbols referenced by the
    solver and is used for model extraction and blocking.
    """

    solver: z3.Solver
    z3_constants: ContractionZ3Constants


@dataclass
class ConstraintPayload:
    """
    A container for Z3 data passed to worker processes.
    `z3_smt2` is SMT-2 string of all formulas asserted in the ConstraintSet.solver.
    `z3_constants_meta` is serialized ConstraintSet.z3_constants.
    """

    z3_smt2: str
    z3_constants_meta: dict


class ConstraintGenerator(ABC):
    """
    Describes how to generate constraints and produce tuning candidates
    for a specific type of tunable problem.

    Implementations of ConstraintGenerator are responsible for encapsulating
    problem-specific information—such as contraction dimensions, sizes, operand types—
    and using that information to generate valid configurations that satisfy the
    constraints imposed by the codegen pipeline and target architecture.

    The `generate_solutions` method returns an iterator over lists of
    `TuningConfiguration` instances. Each list represents a self-contained tuning
    candidate that can be applied to the dispatch root op.

    Example output:
        [
            TuningConfiguration(name="compilation_info", configuration=CompilationInfoAttr(...)),
            TuningConfiguration(name="decomposition_config", configuration=DecompositionConfigAttr(...)),
        ]
    """

    @abstractmethod
    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        """
        Generate a sequence of tuning configuration entries.

        Each constraint generator implementation is responsible for a specific
        codegen pipeline, so the pipeline is implicit in the generator type.
        """
        pass


def solve_z3_contraint_payload(
    constraint_payload: ConstraintPayload,
) -> list[ContractionZ3Assignment]:
    """
    Function executed in worker processes to solve an independent constraint set.
    """
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    solver.add(z3.parse_smt2_string(constraint_payload.z3_smt2, ctx=ctx))
    z3_constants = ContractionZ3Constants.from_meta(
        constraint_payload.z3_constants_meta, ctx
    )
    solution_iter = get_z3_solutions(
        ConstraintSet(solver=solver, z3_constants=z3_constants)
    )

    return list(solution_iter)


def get_z3_solutions(
    constraint_set: ConstraintSet,
) -> Iterator[ContractionZ3Assignment]:
    solver = constraint_set.solver
    z3_constants = constraint_set.z3_constants
    z3_symbols = constraint_set.z3_constants.symbols

    while solver.check() == z3.sat:
        model = solver.model()
        z3_assignment = z3_constants.extract(model)

        # Add new constraints to find the next solution.
        solver.add(
            z3.Or([v != model.eval(v, model_completion=True) for v in z3_symbols])
        )

        yield z3_assignment
