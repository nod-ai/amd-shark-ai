# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import z3  # type: ignore
import math
from typing import Iterator, Optional

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from .. import common, constraint_generator, dispatch_parser, process_utils
from . import rocm_common, rocm_dispatch_constraints


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
) -> list[constraint_generator.ConstraintSet]:
    z3_constants = constraint_generator.ContractionZ3Constants.from_sizes(matmul_size)

    mma_intrinsic_constraints_list = (
        rocm_dispatch_constraints.get_mma_intrinsic_constraints_list(
            lhs_type,
            rhs_type,
            res_type,
            z3_constants.intrinsic_mn,
            z3_constants.intrinsic_mn,
            z3_constants.intrinsic_k,
            gpu_target_info.mma_intrinsics,
        )
    )

    match codegen_pipeline:
        case iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute:
            constraints = (
                rocm_dispatch_constraints.generate_vector_distribute_constraints(
                    matmul_size,
                    lhs_type,
                    rhs_type,
                    res_type,
                    [z3_constants.m_vals, z3_constants.n_vals, z3_constants.k_vals],
                    num_subgroups,
                    z3_constants.subgroup_size,
                    [z3_constants.intrinsic_mn, z3_constants.intrinsic_k],
                    [z3_constants.wg_x, z3_constants.wg_y, z3_constants.wg_z],
                    z3_constants.sg_m_cnt,
                    z3_constants.sg_n_cnt,
                    gpu_target_info,
                    dispatch_kind,
                )
            )
            constraints += [
                v == 0
                for v in z3_constants.subgroup_m_vals + z3_constants.subgroup_n_vals
            ]
        case iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse:
            constraints = rocm_dispatch_constraints.generate_tile_and_fuse_constraints(
                matmul_size,
                lhs_type,
                rhs_type,
                res_type,
                [
                    z3_constants.m_vals,
                    z3_constants.n_vals,
                    z3_constants.k_vals,
                    z3_constants.subgroup_m_vals,
                    z3_constants.subgroup_n_vals,
                ],
                num_subgroups,
                z3_constants.subgroup_size,
                [z3_constants.intrinsic_mn, z3_constants.intrinsic_k],
                [z3_constants.wg_x, z3_constants.wg_y, z3_constants.wg_z],
                z3_constants.sg_m_cnt,
                z3_constants.sg_n_cnt,
                gpu_target_info,
            )

    mega_constraints_list = []
    # Split search space by different mma intrinsic layouts.
    for i, mma_intrinsic_constraints in enumerate(
        mma_intrinsic_constraints_list, start=1
    ):
        solver = z3.Solver()
        final_constraints = constraints.copy() + [mma_intrinsic_constraints]
        solver.add(z3.simplify(z3.And(final_constraints)))
        mega_constraints_list.append(
            constraint_generator.ConstraintSet(solver, z3_constants)
        )

    return mega_constraints_list


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
    pipeline_options_search_space: rocm_dispatch_constraints.PipelineOptionsSearchSpace = rocm_dispatch_constraints.PipelineOptionsSearchSpace(),
    igemm_details: Optional[iree_codegen.IGEMMGenericConvDetails] = None,
    conv_to_igemm_info: Optional[rocm_common.ConvToIgemmInfo] = None,
) -> Iterator[list[common.TuningConfiguration]]:
    # Set use_igemm_convolution for IGEMM convolutions.
    if (
        codegen_pipeline == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
        and dispatch_kind == common.DispatchKind.conv
    ):
        pipeline_options_search_space.use_igemm_convolution = [True]

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

    mega_constraints_list: list[
        constraint_generator.ConstraintSet
    ] = generate_generic_contraction_z3_constraints(
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

    constraint_payload_list = [
        constraint_generator.ConstraintPayload(
            mega_constraints.solver.to_smt2(), mega_constraints.z3_constants.to_meta()
        )
        for mega_constraints in mega_constraints_list
    ]
    tuner_ctx.logger.debug(
        f"Will generate [{len(constraint_payload_list)}] constraint solvers."
    )
    print(f"Running {len(constraint_payload_list)} worker processes...")
    executor = process_utils.MultiprocessExecutor(
        num_workers=len(constraint_payload_list),
    )
    z3_solutions_list = executor.run(
        task_list=constraint_payload_list,
        worker_fn=constraint_generator.solve_z3_contraint_payload,
    )
    tuner_ctx.logger.debug(
        f"Search space sizes for each constraint set: {[len(i) for i in z3_solutions_list]}."
    )
    z3_solutions_list = [x for sublist in z3_solutions_list for x in sublist]

    for z3_assignment in z3_solutions_list:
        intrinsic_mnk_shape = (
            z3_assignment.intrinsic_mn,
            z3_assignment.intrinsic_mn,
            z3_assignment.intrinsic_k,
        )
        mma_attr = rocm_dispatch_constraints.getMMAAttr(
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
            z3_assignment.m_vals,
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            contraction_dims.n,
            z3_assignment.n_vals,
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
            z3_assignment.subgroup_m_vals,
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            contraction_dims.n,
            z3_assignment.subgroup_n_vals,
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
            z3_assignment.k_vals,
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
                padding_conv = rocm_common.get_padding_conv_sizes(
                    bounds,
                    padding_tile_sizes,
                    igemm_iterator_types,
                    conv_to_igemm_info,
                )
        # Setting subgroup basis.
        # TODO(Bangtian): Sync changes from IREE PR: https://github.com/iree-org/iree/pull/22000.
        subgroup_basis_counts = [1] * num_loops
        m_dim = contraction_dims.m[-1]
        subgroup_basis_counts[m_dim] = z3_assignment.sg_m_cnt
        n_dim = contraction_dims.n[-1]
        subgroup_basis_counts[n_dim] = z3_assignment.sg_n_cnt
        subgroup_basis_mapping = list(range(num_loops))

        compilation_infos = rocm_dispatch_constraints.generate_compilation_infos(
            tuner_ctx,
            mma_attr,
            workgroup_tile_sizes,
            reduction_tile_sizes,
            subgroup_tile_sizes,
            (z3_assignment.wg_x, z3_assignment.wg_y, z3_assignment.wg_z),
            z3_assignment.subgroup_size,
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
            if codegen_pipeline in (
                iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
                iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
            ):
                knob_assignment = rocm_common.LLVMGPUContractionKnobs(
                    M=int(math.prod(M)),
                    N=int(math.prod(N)),
                    K=int(math.prod(K)),
                    tile_m=int(math.prod(z3_assignment.m_vals)),
                    tile_n=int(math.prod(z3_assignment.n_vals)),
                    tile_k=int(math.prod(z3_assignment.k_vals)),
                    wg_x=z3_assignment.wg_x,
                    wg_y=z3_assignment.wg_y,
                    wg_z=z3_assignment.wg_z,
                    subgroup_m_cnt=z3_assignment.sg_m_cnt,
                    subgroup_n_cnt=z3_assignment.sg_n_cnt,
                    intrinsic_mn=z3_assignment.intrinsic_mn,
                    intrinsic_k=z3_assignment.intrinsic_k,
                    subgroup_m=int(math.prod(z3_assignment.subgroup_m_vals)),
                    subgroup_n=int(math.prod(z3_assignment.subgroup_n_vals)),
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
    pipeline_options_search_space: rocm_dispatch_constraints.PipelineOptionsSearchSpace = rocm_dispatch_constraints.PipelineOptionsSearchSpace(),
) -> Iterator[list[common.TuningConfiguration]]:
    if (
        dispatch_kind != common.DispatchKind.attention
        or codegen_pipeline
        != iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    ):
        return

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
    constraints = (
        rocm_dispatch_constraints.generate_attention_vector_distribute_constraints(
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
        qk_mma_attr = rocm_dispatch_constraints.getMMAAttr(
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
        pv_mma_attr = rocm_dispatch_constraints.getMMAAttr(
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

        decomposition_config = rocm_common.get_attention_decomposition_config(
            tuner_ctx, qk_lowering_config, pv_lowering_config
        )

        workgroup_size = lookup(sg_m_cnt) * lookup(sg_n_cnt) * lookup(subgroup_size)

        # Set prefetch_num_stages based on whether layouts match.
        # 0/1 = disable prefetching, 2 = two-stage pipeline (default),
        # 3 = three-stage pipeline (separate read, write, compute stages).
        layouts_match = bool(model[can_reuse_qk_output_for_pv_input])
        pipeline_options_search_space.prefetch_num_stages = [2 if layouts_match else 0]

        promote_operands = [0, 1, 2]
        compilation_infos = rocm_dispatch_constraints.generate_compilation_infos(
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
