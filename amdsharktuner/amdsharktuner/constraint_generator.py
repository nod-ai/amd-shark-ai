# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import z3  # type: ignore
import math
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Protocol, Callable
from dataclasses import dataclass

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu, linalg  # type: ignore

from . import common, dispatch_constraints, dispatch_parser


class Z3Vals(Protocol):
    """Marker base class for values extracted from Z3 model."""


class Z3Vars(Protocol):
    """Marker base class for Z3 variables."""

    @property
    def all_vars(self) -> list[z3.ExprRef]:
        """All Z3 variables that participate in this problem."""
        pass

    def eval(self, model: z3.ModelRef) -> Z3Vals:
        """Convert Z3 variables in the given model to values."""
        pass


@dataclass
class ContractionZ3Vals(Z3Vals):
    m_vals: list[int]
    n_vals: list[int]
    k_vals: list[int]
    subgroup_m_vals: list[int]
    subgroup_n_vals: list[int]

    subgroup_size: int
    intrinsic_mn: int
    intrinsic_k: int
    wg_x: int
    wg_y: int
    wg_z: int
    sg_m_cnt: int
    sg_n_cnt: int


@dataclass
class ContractionZ3Vars(Z3Vars):
    m_vars: list[z3.ExprRef]
    n_vars: list[z3.ExprRef]
    k_vars: list[z3.ExprRef]
    subgroup_m_vars: list[z3.ExprRef]
    subgroup_n_vars: list[z3.ExprRef]

    subgroup_size: z3.ExprRef
    intrinsic_mn: z3.ExprRef
    intrinsic_k: z3.ExprRef
    wg_x: z3.ExprRef
    wg_y: z3.ExprRef
    wg_z: z3.ExprRef
    sg_m_cnt: z3.ExprRef
    sg_n_cnt: z3.ExprRef

    @property
    def all_vars(self) -> list[z3.ExprRef]:
        return (
            self.m_vars
            + self.n_vars
            + self.k_vars
            + [
                self.subgroup_size,
                self.intrinsic_mn,
                self.intrinsic_k,
                self.wg_x,
                self.wg_y,
                self.wg_z,
                self.sg_m_cnt,
                self.sg_n_cnt,
            ]
        )

    def eval(self, model: z3.ModelRef) -> ContractionZ3Vals:
        get = lambda v: model[v].as_long()
        return ContractionZ3Vals(
            m_vals=[get(v) for v in self.m_vars],
            n_vals=[get(v) for v in self.n_vars],
            k_vals=[get(v) for v in self.k_vars],
            subgroup_m_vals=[get(v) for v in self.subgroup_m_vars],
            subgroup_n_vals=[get(v) for v in self.subgroup_n_vars],
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
class Z3Solver:
    solver: z3.Solver
    z3_vars: Z3Vars


def adjust_problem_size_for_pipeline(
    contraction_dims: common.ContractionDimensions,
    matmul_size: common.ContractionSizes,
    dispatch_kind: common.DispatchKind,
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
    igemm_details: Optional[iree_codegen.IGEMMGenericConvDetails] = None,
):
    # Adjustment is only needed for IGEMM. Fail if the problem is not a conv
    # going down the TileAndFuse pipeline.
    if (
        codegen_pipeline != iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
        or dispatch_kind != common.DispatchKind.conv
    ):
        return

    pipeline_options_search_space.use_igemm_convolution = [True]

    # Use IGEMM binding details if available for accurate dimension mapping.
    if igemm_details:
        igemm_maps = [
            map_attr.value for map_attr in igemm_details.igemm_contraction_maps
        ]
        igemm_contraction_dims = linalg.infer_contraction_dimensions_from_maps(
            igemm_maps
        )
        assert (
            igemm_contraction_dims
        ), "Failed to infer contraction dimensions from IGEMM maps"

        bounds = list(igemm_details.igemm_loop_bounds)

        # Update contraction_dims with IGEMM structure.
        contraction_dims.m = list(igemm_contraction_dims.m)
        contraction_dims.n = list(igemm_contraction_dims.n)
        contraction_dims.k = list(igemm_contraction_dims.k)
        contraction_dims.batch = list(igemm_contraction_dims.batch)

        # Update matmul_size with IGEMM loop bounds (K is already flattened!).
        matmul_size.M = [bounds[i] for i in contraction_dims.m]
        matmul_size.N = [bounds[i] for i in contraction_dims.n]
        matmul_size.K = [bounds[i] for i in contraction_dims.k]
        matmul_size.B = [bounds[i] for i in contraction_dims.batch]
        return

    # Fallback: Manual flattening for legacy path when IGEMM details are unavailable.
    # TODO(Bangtian): Once all IGEMM implementation is complete, fully remove this fallback path
    # and corresponding tests.
    contraction_dims.k = [contraction_dims.k[0]]
    matmul_size.K = [math.prod(matmul_size.K)]


def make_contraction_z3_vars(
    matmul_size: common.ContractionSizes,
) -> ContractionZ3Vars:
    M, N, K = matmul_size.M, matmul_size.N, matmul_size.K

    m_vars = [z3.Int(f"m{i}") for i in range(len(M))]
    n_vars = [z3.Int(f"n{i}") for i in range(len(N))]
    k_vars = [z3.Int(f"k{i}") for i in range(len(K))]
    subgroup_m_vars = [z3.Int(f"subgroup_m{i}") for i in range(len(M))]
    subgroup_n_vars = [z3.Int(f"subgroup_n{i}") for i in range(len(N))]

    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = z3.Int("wg_x"), z3.Int("wg_y"), z3.Int("wg_z")
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")

    return ContractionZ3Vars(
        m_vars=m_vars,
        n_vars=n_vars,
        k_vars=k_vars,
        subgroup_m_vars=subgroup_m_vars,
        subgroup_n_vars=subgroup_n_vars,
        subgroup_size=subgroup_size,
        intrinsic_mn=intrinsic_mn,
        intrinsic_k=intrinsic_k,
        wg_x=wg_x,
        wg_y=wg_y,
        wg_z=wg_z,
        sg_m_cnt=sg_m_cnt,
        sg_n_cnt=sg_n_cnt,
    )


def generate_generic_contraction_z3_constraints(
    tuner_ctx: common.TunerContext,
    gpu_target_info: iree_gpu.TargetInfo,
    dispatch_kind: common.DispatchKind,
    matmul_size: common.ContractionSizes,
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    num_subgroups: int = 4,
) -> Z3Solver:
    z3_vars = make_contraction_z3_vars(matmul_size)
    solver = z3.Solver()
    match codegen_pipeline:
        case iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute:
            constraints = dispatch_constraints.generate_vector_distribute_constraints(
                matmul_size,
                lhs_type,
                rhs_type,
                res_type,
                [z3_vars.m_vars, z3_vars.n_vars, z3_vars.k_vars],
                num_subgroups,
                z3_vars.subgroup_size,
                [z3_vars.intrinsic_mn, z3_vars.intrinsic_k],
                [z3_vars.wg_x, z3_vars.wg_y, z3_vars.wg_z],
                z3_vars.sg_m_cnt,
                z3_vars.sg_n_cnt,
                gpu_target_info,
                dispatch_kind,
            )
            constraints += [
                v == 0 for v in z3_vars.subgroup_m_vars + z3_vars.subgroup_n_vars
            ]
        case iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse:
            constraints = dispatch_constraints.generate_tile_and_fuse_constraints(
                matmul_size,
                lhs_type,
                rhs_type,
                res_type,
                [
                    z3_vars.m_vars,
                    z3_vars.n_vars,
                    z3_vars.k_vars,
                    z3_vars.subgroup_m_vars,
                    z3_vars.subgroup_n_vars,
                ],
                num_subgroups,
                z3_vars.subgroup_size,
                [z3_vars.intrinsic_mn, z3_vars.intrinsic_k],
                [z3_vars.wg_x, z3_vars.wg_y, z3_vars.wg_z],
                z3_vars.sg_m_cnt,
                z3_vars.sg_n_cnt,
                gpu_target_info,
            )

    solver.add(z3.simplify(z3.And(constraints)))
    tuner_ctx.logger.debug(f"Initial constraints: {solver}")

    return Z3Solver(solver, z3_vars)


def get_z3_solutions(z3_solver: Z3Solver) -> Iterator[Z3Vals]:
    solver = z3_solver.solver
    z3_vars = z3_solver.z3_vars
    z3_all_vars = z3_solver.z3_vars.all_vars

    while solver.check() == z3.sat:
        model = solver.model()

        z3_vals = z3_vars.eval(model)

        # Add new constraints to find the next solution.
        solver.add(z3.Or([v != model[v] for v in z3_all_vars]))

        yield z3_vals


def generate_generic_contraction_solutions(
    tuner_ctx: common.TunerContext,
    gpu_target_info: iree_gpu.TargetInfo,
    contraction_dims: common.ContractionDimensions,
    matmul_size: common.ContractionSizes,
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    dispatch_kind: common.DispatchKind,
    indexing_maps: list[ir.AffineMap],
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    num_subgroups: int = 4,
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace = dispatch_constraints.PipelineOptionsSearchSpace(),
    igemm_details: Optional[iree_codegen.IGEMMGenericConvDetails] = None,
    conv_to_igemm_info: Optional[common.ConvToIgemmInfo] = None,
) -> Iterator[list[common.TuningConfiguration]]:
    adjust_problem_size_for_pipeline(
        contraction_dims,
        matmul_size,
        dispatch_kind,
        pipeline_options_search_space,
        codegen_pipeline,
        igemm_details,
    )

    # Apply padding for TileAndFuse pipeline to get better tile sizes.
    overpadding_applied = False
    if codegen_pipeline == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse:
        # Use IGEMM maps if available (dimensions were restructured), otherwise use original indexing maps.
        padding_maps = indexing_maps
        if igemm_details:
            padding_maps = [
                map_attr.value for map_attr in igemm_details.igemm_contraction_maps
            ]

        (
            matmul_size.M,
            matmul_size.N,
            overpadding_applied,
        ) = common.calculate_padded_dimensions(
            matmul_size.M, matmul_size.N, contraction_dims, padding_maps
        )

    M, N, K = matmul_size.M, matmul_size.N, matmul_size.K
    tuner_ctx.logger.debug(
        f"M={M}, N={N}, K={K}, overpadding_applied={overpadding_applied}"
    )

    constraints = generate_generic_contraction_z3_constraints(
        tuner_ctx,
        gpu_target_info,
        dispatch_kind,
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        codegen_pipeline,
        num_subgroups=num_subgroups,
    )

    num_loops = (
        len(contraction_dims.m)
        + len(contraction_dims.n)
        + len(contraction_dims.k)
        + len(contraction_dims.batch)
    )

    z3_solutions_iter: Iterator[ContractionZ3Vals] = get_z3_solutions(constraints)

    for z3_vals in list(z3_solutions_iter):
        intrinsic_mnk_shape = (
            z3_vals.intrinsic_mn,
            z3_vals.intrinsic_mn,
            z3_vals.intrinsic_k,
        )
        mma_attr = dispatch_constraints.getMMAAttr(
            res_type.element_type,
            *intrinsic_mnk_shape,
            lhs_type.element_type,
            rhs_type.element_type,
            gpu_target_info.mma_intrinsics,
        )

        # Check if any dimension requires padding to align with intrinsic sizes.
        required_padding = any(
            p[-1] % i != 0 for p, i in zip((M, N, K), intrinsic_mnk_shape, strict=True)
        )
        if required_padding:
            tuner_ctx.logger.debug(
                f"Required padding detected: M={M}, N={N}, K={K}, intrinsic_shape={intrinsic_mnk_shape}"
            )

        def set_cdim_tile_sizes(tile_sizes, contraction_dims, csizes):
            for dim, size in zip(contraction_dims, csizes):
                tile_sizes[dim] = size

        # Get workgroup tile sizes.
        workgroup_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            contraction_dims.m,
            z3_vals.m_vals,
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            contraction_dims.n,
            z3_vals.n_vals,
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            contraction_dims.batch,
            [1] * len(contraction_dims.batch),
        )

        # Get subgroup tile sizes.
        subgroup_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            contraction_dims.m,
            z3_vals.subgroup_m_vals,
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            contraction_dims.n,
            z3_vals.subgroup_n_vals,
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            contraction_dims.batch,
            [1] * len(contraction_dims.batch),
        )

        # Get reduction tile sizes.
        reduction_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            reduction_tile_sizes,
            contraction_dims.k,
            z3_vals.k_vals,
        )

        promote_operands = [0, 1]
        padding = None
        padding_conv = None
        if required_padding or overpadding_applied:
            padding_tile_sizes = list(workgroup_tile_sizes)
            for k_dim in contraction_dims.k:
                padding_tile_sizes[k_dim] = reduction_tile_sizes[k_dim]

            mma_intrinsic_k = mma_attr.mnk_shape[2]
            inner_k_dim = contraction_dims.k[-1]
            padding_tile_sizes[inner_k_dim] *= mma_intrinsic_k

            padding = padding_tile_sizes

            # Calculate padding_conv sizes for convolutions when using IGEMM.
            if conv_to_igemm_info and igemm_details:
                # Use IGEMM loop bounds directly from igemm_details.
                bounds = list(igemm_details.igemm_loop_bounds)
                igemm_iterator_types = [
                    str(it) for it in igemm_details.igemm_loop_iterators
                ]
                padding_conv = common.get_padding_conv_sizes(
                    bounds,
                    padding_tile_sizes,
                    igemm_iterator_types,
                    conv_to_igemm_info,
                )
        # Setting subgroup basis.
        # TODO(Bangtian): Sync changes from IREE PR: https://github.com/iree-org/iree/pull/22000.
        subgroup_basis_counts = [1] * num_loops
        m_dim = contraction_dims.m[-1]
        subgroup_basis_counts[m_dim] = z3_vals.sg_m_cnt
        n_dim = contraction_dims.n[-1]
        subgroup_basis_counts[n_dim] = z3_vals.sg_n_cnt
        subgroup_basis_mapping = list(range(num_loops))

        compilation_infos = dispatch_constraints.generate_compilation_infos(
            tuner_ctx,
            mma_attr,
            workgroup_tile_sizes,
            reduction_tile_sizes,
            subgroup_tile_sizes,
            (z3_vals.wg_x, z3_vals.wg_y, z3_vals.wg_z),
            z3_vals.subgroup_size,
            subgroup_basis_counts,
            subgroup_basis_mapping,
            promote_operands,
            codegen_pipeline,
            pipeline_options_search_space,
            allowed_waves_per_eu,
            padding=padding,
            padding_conv=padding_conv,
        )

        knob_assignment = None
        for compilation_info in compilation_infos:
            if (
                codegen_pipeline
                == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
            ):
                knob_assignment = common.LLVMGPUVectorDistributeContractionKnobs(
                    M=int(math.prod(M)),
                    N=int(math.prod(N)),
                    K=int(math.prod(K)),
                    tile_m=workgroup_tile_sizes[0],
                    tile_n=workgroup_tile_sizes[1],
                    tile_k=reduction_tile_sizes[2],
                    wg_x=z3_vals.wg_x,
                    wg_y=z3_vals.wg_y,
                    wg_z=z3_vals.wg_z,
                    subgroup_m_cnt=z3_vals.sg_m_cnt,
                    subgroup_n_cnt=z3_vals.sg_n_cnt,
                    intrinsic_mn=z3_vals.intrinsic_mn,
                    intrinsic_k=z3_vals.intrinsic_k,
                    subgroup_m=subgroup_tile_sizes[0],
                    subgroup_n=subgroup_tile_sizes[1],
                    subgroup_k=subgroup_tile_sizes[2],
                )
            yield [
                common.TuningConfiguration(
                    name="compilation_info",
                    configuration=compilation_info,
                    knob_assignment=knob_assignment,
                )
            ]


def generate_attention_solutions(
    tuner_ctx: common.TunerContext,
    gpu_target_info: iree_gpu.TargetInfo,
    op_info: dispatch_parser.AttentionOpInfo,
    dispatch_kind: common.DispatchKind,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    num_subgroups: int = 4,
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace = dispatch_constraints.PipelineOptionsSearchSpace(),
) -> Iterator[list[common.TuningConfiguration]]:
    if (
        dispatch_kind != common.DispatchKind.attention
        or codegen_pipeline
        != iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    ):
        return []

    m_var = z3.Int("m_tile")
    n_var = z3.Int("n_tile")
    k_var = z3.Int("k_tile")

    subgroup_size = z3.Int("subgroup_size")
    qk_intrinsic_mn = z3.Int("qk_intrinsic_mn")
    qk_intrinsic_k = z3.Int("qk_intrinsic_k")
    pv_intrinsic_mn = z3.Int("pv_intrinsic_mn")
    pv_intrinsic_k = z3.Int("pv_intrinsic_k")
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")

    # Used to determine if prefetch_num_stages can be enabled.
    # See: https://github.com/iree-org/iree/blob/411aa64083a2303946b4d2d72d00e6a6814fbafb/compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp#L974-L976.
    can_reuse_qk_output_for_pv_input = z3.Bool("can_reuse_qk_output_for_pv_input")

    all_vars = (
        [m_var]
        + [n_var]
        + [k_var]
        + [
            subgroup_size,
            qk_intrinsic_mn,
            qk_intrinsic_k,
            pv_intrinsic_mn,
            pv_intrinsic_k,
            sg_m_cnt,
            sg_n_cnt,
        ]
    )

    solver = z3.Solver()
    constraints = dispatch_constraints.generate_attention_vector_distribute_constraints(
        op_info.qk_matmul,
        op_info.pv_matmul,
        op_info.transposed_q,
        op_info.transposed_k,
        op_info.transposed_v,
        [m_var, n_var, k_var],
        num_subgroups,
        subgroup_size,
        [qk_intrinsic_mn, qk_intrinsic_k],
        [pv_intrinsic_mn, pv_intrinsic_k],
        sg_m_cnt,
        sg_n_cnt,
        can_reuse_qk_output_for_pv_input,
        gpu_target_info,
    )

    solver.add(z3.simplify(z3.And(constraints)))
    tuner_ctx.logger.debug(f"Initial constraints: {solver}")

    i = 0
    while solver.check() == z3.sat:
        model = solver.model()

        def lookup(var):
            return model[var].as_long()

        qk_intrinsic_mnk_shape = (
            lookup(qk_intrinsic_mn),
            lookup(qk_intrinsic_mn),
            lookup(qk_intrinsic_k),
        )
        qk_mma_attr = dispatch_constraints.getMMAAttr(
            op_info.qk_matmul.acc_type,
            *qk_intrinsic_mnk_shape,
            op_info.qk_matmul.lhs_type,
            op_info.qk_matmul.rhs_type,
            gpu_target_info.mma_intrinsics,
        )

        pv_intrinsic_mnk_shape = (
            lookup(pv_intrinsic_mn),
            lookup(pv_intrinsic_mn),
            lookup(pv_intrinsic_k),
        )
        pv_mma_attr = dispatch_constraints.getMMAAttr(
            op_info.pv_matmul.acc_type,
            *pv_intrinsic_mnk_shape,
            op_info.pv_matmul.lhs_type,
            op_info.pv_matmul.rhs_type,
            gpu_target_info.mma_intrinsics,
        )

        # Get workgroup tile sizes.
        workgroup_tile_sizes = [0] * op_info.domain_rank
        reduction_tile_sizes = [0] * op_info.domain_rank

        for b in op_info.batch_dims:
            workgroup_tile_sizes[b] = 1
        for m in op_info.m_dims[:-1]:
            workgroup_tile_sizes[m] = 1
        for n in op_info.n_dims[:-1]:
            workgroup_tile_sizes[n] = 1
        for k2 in op_info.k2_dims[:-1]:
            reduction_tile_sizes[k2] = 1

        workgroup_tile_sizes[op_info.m_dims[-1]] = lookup(m_var)
        workgroup_tile_sizes[op_info.n_dims[-1]] = lookup(n_var)
        reduction_tile_sizes[op_info.k2_dims[-1]] = lookup(k_var)

        subgroup_basis_counts = [1] * op_info.domain_rank
        subgroup_basis_mapping = list(range(op_info.domain_rank))
        subgroup_basis_counts[op_info.m_dims[-1]] = lookup(sg_m_cnt)
        subgroup_basis_counts[op_info.n_dims[-1]] = lookup(sg_n_cnt)
        qk_basis_mapping = [
            mapping
            for i, mapping in enumerate(subgroup_basis_mapping)
            if i not in op_info.n_dims
        ]
        qk_config = {
            "mma_kind": qk_mma_attr,
            "subgroup_basis": [subgroup_basis_counts, qk_basis_mapping],
            "promote_operands": [0, 1],
        }

        qk_lowering_config = common.get_lowering_config(
            tuner_ctx=tuner_ctx, **qk_config
        )

        pv_basis_mapping = [
            mapping
            for i, mapping in enumerate(subgroup_basis_mapping)
            if i not in op_info.k1_dims
        ]
        pv_config = {
            "mma_kind": pv_mma_attr,
            "subgroup_basis": [subgroup_basis_counts, pv_basis_mapping],
            "promote_operands": [1],
        }
        pv_lowering_config = common.get_lowering_config(
            tuner_ctx=tuner_ctx, **pv_config
        )

        decomposition_config = common.get_attention_decomposition_config(
            tuner_ctx, qk_lowering_config, pv_lowering_config
        )

        workgroup_size = lookup(sg_m_cnt) * lookup(sg_n_cnt) * lookup(subgroup_size)

        # Set prefetch_num_stages based on whether layouts match.
        # 0/1 = disable prefetching, 2 = two-stage pipeline (default),
        # 3 = three-stage pipeline (separate read, write, compute stages).
        layouts_match = bool(model[can_reuse_qk_output_for_pv_input])
        pipeline_options_search_space.prefetch_num_stages = [2 if layouts_match else 0]

        promote_operands = [0, 1, 2]
        compilation_infos = dispatch_constraints.generate_compilation_infos(
            tuner_ctx,
            None,
            workgroup_tile_sizes,
            reduction_tile_sizes,
            [0, 0, 0],
            (workgroup_size, 1, 1),
            lookup(subgroup_size),
            subgroup_basis_counts,
            subgroup_basis_mapping,
            promote_operands,
            codegen_pipeline,
            pipeline_options_search_space,
            allowed_waves_per_eu,
            padding=None,
        )
        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1

        for compilation_info in compilation_infos:
            config_list = [
                common.TuningConfiguration(
                    name="compilation_info", configuration=compilation_info
                ),
                common.TuningConfiguration(
                    name="decomposition_config", configuration=decomposition_config
                ),
            ]
            yield config_list


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
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        """
        Generate a sequence of tuning configuration entries for the specified pipeline.
        """
        pass


class ContractionOpInterfaceConstraintGenerator(ConstraintGenerator):
    def __init__(self, op_info: dispatch_parser.ContractionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=self.op_info.dims,
            matmul_size=self.op_info.matmul_size,
            lhs_type=self.op_info.lhs_type,
            rhs_type=self.op_info.rhs_type,
            res_type=self.op_info.res_type,
            dispatch_kind=common.DispatchKind.contraction,
            indexing_maps=self.op_info.indexing_maps,
            codegen_pipeline=codegen_pipeline,
            **pipeline_constraint_options,
        )


class ConvolutionOpInterfaceConstraintGenerator(ConstraintGenerator):
    def __init__(self, op_info: dispatch_parser.ConvolutionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        # TODO(Bangtian): Simplify the function signature to accept op_info directly instead of
        # unpacking all individual fields.
        return generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=self.op_info.dims,
            matmul_size=self.op_info.matmul_size,
            lhs_type=self.op_info.lhs_type,
            rhs_type=self.op_info.rhs_type,
            res_type=self.op_info.res_type,
            dispatch_kind=common.DispatchKind.conv,
            indexing_maps=self.op_info.indexing_maps,
            codegen_pipeline=codegen_pipeline,
            igemm_details=self.op_info.igemm_details,
            conv_to_igemm_info=self.op_info.conv_to_igemm_info,
            **pipeline_constraint_options,
        )


class AttentionOpInterfaceConstraintGenerator(ConstraintGenerator):
    """
    Constraint generator for the IREE LinalgExt AttentionOp.

    Generates tuning configurations for attention operations.

    Attributes:
        op_info: AttentionOpInfo containing all attention operation metadata.
    """

    def __init__(self, op_info: dispatch_parser.AttentionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return generate_attention_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            op_info=self.op_info,
            dispatch_kind=common.DispatchKind.attention,
            codegen_pipeline=codegen_pipeline,
            **pipeline_constraint_options,
        )
