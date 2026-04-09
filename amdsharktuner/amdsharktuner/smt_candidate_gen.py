# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import Iterator, Optional
from typing_extensions import override

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore
import z3  # type: ignore

from .common import (
    AttrKey,
    CompilationInfoBuilder,
    KnobSymbols,
    SMTKnobAssignment,
)


logger = logging.getLogger("smt_candidate_gen")


def _resolve_knob_array_attr_template(
    template_entry: ir.ArrayAttr,
    knob_assignment: SMTKnobAssignment,
) -> list[int]:
    """Resolve IntKnobAttr placeholders to knob assignment values.
    E.g.
    template_entry = [#iree_codegen.smt.int_knob<"wg_m">,
                      #iree_codegen.smt.int_knob<"wg_n">]
    knob_assignment = {"wg_m": 4, "wg_n": 5}
    result = [4, 5].
    """
    result: list[int] = []
    for elem in template_entry:
        if not isinstance(elem, iree_codegen.IntKnobAttr):
            raise TypeError(f"Unexpected element in array template entry: {elem}")

        assert elem.name in knob_assignment, (
            f"Knob '{elem.name}' not found in assignment.\n"
            f"Available knobs: \n{list(knob_assignment.keys())}"
        )
        result.append(knob_assignment[elem.name])

    return result


def _get_template_entry(
    knob_template: ir.DictAttr,
    attr_key: AttrKey,
) -> Optional[ir.Attribute]:
    """Return the knob entry for `attr_key` if present.

    E.g.
    knob_template = {wg_m = #iree_codegen.smt.int_knob<"wg_m">,
                     mma = #iree_codegen.smt.one_of_knob<"mma", ["a", "b"]>}
    attr_key = AttrKey("wg_n", IntKnobAttr)
    result = None.
    attr_key = AttrKey("mma", OneOfKnobAttr)
    result = #iree_codegen.smt.one_of_knob<"mma", ["a", "b"]>.

    """
    if attr_key.name not in knob_template:
        return None
    template_entry = knob_template[attr_key.name]
    assert isinstance(template_entry, attr_key.attr_type)
    return template_entry


class GPUCompilationInfoBuilder(CompilationInfoBuilder):
    """Key names and builders for GPU compilation info attrs, matching IREE's
    GPULoweringConfigUtils.cpp conventions.
    """

    class LoweringConfig(CompilationInfoBuilder.LoweringConfig):
        """Key names used in iree_gpu.LoweringConfigAttr's DictionaryAttr."""

        # Tiling levels.
        WORKGROUP: AttrKey[ir.ArrayAttr] = AttrKey("workgroup", ir.ArrayAttr)
        REDUCTION: AttrKey[ir.ArrayAttr] = AttrKey("reduction", ir.ArrayAttr)
        THREAD: AttrKey[ir.ArrayAttr] = AttrKey("thread", ir.ArrayAttr)
        SUBGROUP: AttrKey[ir.ArrayAttr] = AttrKey("subgroup", ir.ArrayAttr)

        # MMA intrinsic: OneOfKnobAttr index selects from the options array.
        MMA_KIND: AttrKey[iree_codegen.OneOfKnobAttr] = AttrKey(
            "mma_kind", iree_codegen.OneOfKnobAttr
        )

        # Subgroup basis: stored as [[counts...], [mapping...]] (two i64 arrays).
        SUBGROUP_BASIS: AttrKey[ir.DictAttr] = AttrKey("subgroup_basis", ir.DictAttr)
        SUBGROUP_BASIS_COUNTS: AttrKey[ir.ArrayAttr] = AttrKey("counts", ir.ArrayAttr)
        SUBGROUP_BASIS_MAPPING: AttrKey[ir.ArrayAttr] = AttrKey("mapping", ir.ArrayAttr)

        @classmethod
        def _get_i64_array_attr(cls, vals: list[int]) -> ir.ArrayAttr:
            i64 = ir.IntegerType.get_signless(64)
            return ir.ArrayAttr.get([ir.IntegerAttr.get(i64, v) for v in vals])

        @classmethod
        def _add_tiling_level_config_entry(
            cls,
            knob_template: ir.DictAttr,
            knob_assignment: SMTKnobAssignment,
            config_entries: dict[str, ir.Attribute],
        ) -> None:
            # template_attr: ArrayAttr<IntKnobAttr>.
            for key in (cls.WORKGROUP, cls.REDUCTION, cls.THREAD, cls.SUBGROUP):
                template_attr = _get_template_entry(knob_template, key)
                if not template_attr:
                    continue
                config_entries[key.name] = cls._get_i64_array_attr(
                    _resolve_knob_array_attr_template(template_attr, knob_assignment)
                )

        @classmethod
        def _add_mma_kind_config_entry(
            cls,
            knob_template: ir.DictAttr,
            knob_assignment: SMTKnobAssignment,
            config_entries: dict[str, ir.Attribute],
        ) -> None:
            # MMA kind: OneOfKnobAttr holds options; knob_assignment gives the index.
            # mma_kind_tmpl: OneOfKnobAttr.
            mma_kind_tmpl = _get_template_entry(knob_template, cls.MMA_KIND)
            if not mma_kind_tmpl:
                return
            smt_var_name = mma_kind_tmpl.name
            mma_idx = knob_assignment[smt_var_name]
            config_entries[cls.MMA_KIND.name] = mma_kind_tmpl.options[mma_idx]

        @classmethod
        def _add_subgroup_basis_config_entry(
            cls,
            knob_template: ir.DictAttr,
            knob_assignment: SMTKnobAssignment,
            config_entries: dict[str, ir.Attribute],
        ) -> None:
            # Subgroup basis: stored as [[counts...], [mapping...]] in the config.
            # subgroup_basis_tmpl: DictAttr.
            # counts_tmpl: ArrayAttr<IntKnobAttr>.
            # mapping_tmpl: ArrayAttr<IntKnobAttr>.
            subgroup_basis_tmpl = _get_template_entry(knob_template, cls.SUBGROUP_BASIS)
            if not subgroup_basis_tmpl:
                return
            counts_tmpl = _get_template_entry(
                subgroup_basis_tmpl, cls.SUBGROUP_BASIS_COUNTS
            )
            mapping_tmpl = _get_template_entry(
                subgroup_basis_tmpl, cls.SUBGROUP_BASIS_MAPPING
            )
            if not counts_tmpl:
                return
            counts = _resolve_knob_array_attr_template(counts_tmpl, knob_assignment)
            if not mapping_tmpl:
                # Default mapping: [0, 1, ...].
                mapping = list(range(len(counts)))
            else:
                mapping = _resolve_knob_array_attr_template(
                    mapping_tmpl, knob_assignment
                )
            config_entries[cls.SUBGROUP_BASIS.name] = ir.ArrayAttr.get(
                [cls._get_i64_array_attr(counts), cls._get_i64_array_attr(mapping)]
            )

        @classmethod
        def build_lowering_config_attr(
            cls,
            constraints_op: iree_codegen.ConstraintsOp,
            knob_assignment: SMTKnobAssignment,
        ) -> iree_gpu.LoweringConfigAttr:
            knob_template: ir.DictAttr = constraints_op.knobs
            config_entries: dict[
                str, ir.Attribute
            ] = {}  # Built up and passed to LoweringConfigAttr.

            cls._add_tiling_level_config_entry(
                knob_template, knob_assignment, config_entries
            )
            cls._add_mma_kind_config_entry(
                knob_template, knob_assignment, config_entries
            )
            cls._add_subgroup_basis_config_entry(
                knob_template, knob_assignment, config_entries
            )

            return iree_gpu.LoweringConfigAttr.get(ir.DictAttr.get(config_entries))

    class TranslationInfo(CompilationInfoBuilder.TranslationInfo):
        """Key names in the knobs dict for iree_codegen.TranslationInfoAttr."""

        WORKGROUP_SIZE: AttrKey[ir.ArrayAttr] = AttrKey("workgroup_size", ir.ArrayAttr)
        SUBGROUP_SIZE: AttrKey[iree_codegen.IntKnobAttr] = AttrKey(
            "subgroup_size", iree_codegen.IntKnobAttr
        )

        @classmethod
        def _resolve_workgroup_size(
            cls,
            knob_template: ir.DictAttr,
            knob_assignment: SMTKnobAssignment,
        ) -> Optional[list[int]]:
            workgroup_size_tmpl = _get_template_entry(knob_template, cls.WORKGROUP_SIZE)
            if not workgroup_size_tmpl:
                return None
            return _resolve_knob_array_attr_template(
                workgroup_size_tmpl, knob_assignment
            )

        @classmethod
        def _resolve_subgroup_size(
            cls,
            knob_template: ir.DictAttr,
            knob_assignment: SMTKnobAssignment,
        ) -> Optional[int]:
            subgroup_size_tmpl = _get_template_entry(knob_template, cls.SUBGROUP_SIZE)
            if not subgroup_size_tmpl:
                return None
            return knob_assignment[subgroup_size_tmpl.name]

        @classmethod
        def build_translation_info_attr(
            cls,
            constraints_op: iree_codegen.ConstraintsOp,
            knob_assignment: SMTKnobAssignment,
        ) -> iree_codegen.TranslationInfoAttr:

            knob_template: ir.DictAttr = constraints_op.knobs

            workgroup_size = cls._resolve_workgroup_size(knob_template, knob_assignment)
            subgroup_size = cls._resolve_subgroup_size(knob_template, knob_assignment)

            pipeline: ir.Attribute = constraints_op.pipeline

            return iree_codegen.TranslationInfoAttr.get(
                pipeline,
                workgroup_size=workgroup_size,
                subgroup_size=subgroup_size,
            )

    @override
    @classmethod
    def build_compilation_info_attr(
        cls,
        constraints_op: iree_codegen.ConstraintsOp,
        knob_assignment: SMTKnobAssignment,
    ) -> iree_codegen.CompilationInfoAttr:
        lowering_config = cls.LoweringConfig.build_lowering_config_attr(
            constraints_op, knob_assignment
        )
        translation_info = cls.TranslationInfo.build_translation_info_attr(
            constraints_op, knob_assignment
        )
        return iree_codegen.CompilationInfoAttr.get(lowering_config, translation_info)


def get_z3_assignment_from_model(
    model: z3.ModelRef,
    z3_const_exprs: KnobSymbols,
) -> SMTKnobAssignment:
    def get_z3_const_val(v: z3.ExprRef) -> int:
        # Evaluate arbitrary expressions over a model, convert z3 expr to int.
        val = model.eval(v)
        assert z3.is_int_value(
            val
        ), f"Unassigned or non-concrete constant: {v} -> {val}"
        return val.as_long()

    return SMTKnobAssignment(
        {name: get_z3_const_val(expr) for name, expr in z3_const_exprs.items()}
    )


def get_knobs_from_constraint_op(
    constraints_op: iree_codegen.ConstraintsOp,
    z3_ctx: z3.Context,
) -> KnobSymbols:
    """Extract knob names from a ConstraintsOp and return z3 Int constants.

    Recursively walks the knobs DictAttr, collecting the name of every
    IntKnobAttr and OneOfKnobAttr leaf. Returns one z3 Int constant per name,
    consistent with the declarations in `convert_constraints_op_to_smtlib`.

    Example:
    given knobs = {workgroup_size = #iree_codegen.smt.int_knob<"wg_m">,
                   mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", ["a", "b"]>}
                   subgroup_basis = {
                       counts = [#iree_codegen.smt.int_knob<"sg_x">,
                                 #iree_codegen.smt.int_knob<"sg_y">],
                       mapping = [#iree_codegen.smt.int_knob<"map_0">,
                                  #iree_codegen.smt.int_knob<"map_1">]
                    }
    returns
    KnobSymbols(
        {
            "wg_m": z3.Int("wg_m"),
            "mma_idx": z3.Int("mma_idx")
            "sg_x": z3.Int("sg_x"),
            "sg_y": z3.Int("sg_y"),
            "map_0": z3.Int("map_0"),
            "map_1": z3.Int("map_1"),
        }
    ).
    """
    knob_names: list[str] = []

    def collect(attr: ir.Attribute) -> None:
        match attr:
            case iree_codegen.IntKnobAttr() | iree_codegen.OneOfKnobAttr():
                knob_names.append(attr.name)
            case ir.ArrayAttr():
                for elem in attr:
                    collect(elem)
            case ir.DictAttr():
                for entry in attr:
                    collect(entry.attr)
            case _:
                raise TypeError(f"Unknown knob attribute type: {type(attr)}")

    collect(constraints_op.knobs)

    return KnobSymbols({name: z3.Int(name, ctx=z3_ctx) for name in knob_names})


def generate_solutions_from_constraint_op(
    constraints_op: iree_codegen.ConstraintsOp,
    z3_ctx: z3.Context,
) -> Iterator[SMTKnobAssignment]:
    smtlib = iree_codegen.convert_constraints_op_to_smtlib(
        constraints_op, emit_reset=False
    )
    # Prevent solving hangs.
    if "(reset)" in smtlib:
        raise RuntimeError(f"Unexpected reset string in SMTLIB: \n{smtlib}")

    z3_const_exprs = get_knobs_from_constraint_op(constraints_op, z3_ctx)
    z3_vars = list(z3_const_exprs.values())

    solver = z3.Solver(ctx=z3_ctx)
    solver.add(z3.parse_smt2_string(smtlib, ctx=z3_ctx))

    count = 0
    while solver.check() == z3.sat:
        model = solver.model()

        # Add new constraints to find the next solution.
        solver.add(z3.Or([v != model.eval(v, model_completion=True) for v in z3_vars]))

        z3_assignment = get_z3_assignment_from_model(model, z3_const_exprs)
        count += 1
        logger.debug(f"Solution #{count}: {z3_assignment}")
        yield z3_assignment
