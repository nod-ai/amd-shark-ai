# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bijective-parity regression coverage for the compiler-bridge attention
path against the legacy Python Z3 attention generator.

The legacy `rocm_solutions.generate_attention_solutions` was removed in
the IREE-constraints refactor, so this test compares against a *snapshot*
of its candidate set checked in at
`model_tuner/attn_parity_python.txt`. That snapshot was generated on
branch `attn-parity-old-snapshot` (pinned at commit abb556619a, the
parent of the IREE-constraints refactor) by
`model_tuner/parity_check.py --side=python`. The companion
`model_tuner/attn_parity_results.txt` documents the per-MMA-pair
breakdown and reproduction command.

This test:
  1. Runs the compiler-bridge path end-to-end on the same fixture
     shape (2x128x64x128x64 attention on gfx942 with two F16 MMAs).
  2. Extracts the semantic signature
     `(qk_mma, pv_mma, workgroup_tile, reduction_tile)` from each
     candidate produced.
  3. Asserts the set of signatures equals the set extracted from the
     python snapshot — i.e. bijective parity.

The signature comparison is what catches regressions: any constraint
that drops a python candidate or admits an extra one will breach
equality. (Whole-string equality is too strict — the compiler emits
configs in a structurally different form than the python tuner, e.g.
nesting `decomposition_config` inside `lowering_config` and adding
`smt_internal_*` aux entries — but the semantic content matches.)

When the python-side enumeration changes (which happens only when the
IREE-side emitter intentionally tracks a new variant), regenerate the
snapshot via `model_tuner/parity_check.py --side=python` on the
snapshot branch and commit the new `attn_parity_python.txt`. Re-running
the test will then succeed against the new reference.
"""

import re
from pathlib import Path

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from amdsharktuner import candidate_gen, common
from amdsharktuner.rocm.rocm_tuners import ROCmAttentionVectorDistributeTuner


# Small attention shape (2 x 128 x 64 x 128 x 64) with two F16 MMAs in
# the target. Chosen to keep the v0 candidate count manageable while still
# exercising both qk_mma_idx and pv_mma_idx selection paths.
_FIXTURE_MLIR = """
module attributes {hal.device.targets = [
    #hal.device.target<"hip", [#hal.executable.target<"rocm", "rocm-hsaco-fb",
        {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
            features = "", wgp = <compute = fp32, storage = b32,
            subgroup = shuffle,
            mma = [<MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>],
            subgroup_size_choices = [64], max_load_instruction_bits = 128,
            max_workgroup_sizes = [1024, 1024, 1024],
            max_thread_count_per_workgroup = 1024,
            max_workgroup_memory_bytes = 65536,
            max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>]> :
                !hal.device]} {
  func.func @attention_bridge(
      %q: tensor<2x128x64xf16>, %k: tensor<2x128x64xf16>,
      %v: tensor<2x128x64xf16>, %scale: f16) -> tensor<2x128x64xf16>
      attributes {hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
          {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
              features = "", wgp = <compute = fp32, storage = b32,
              subgroup = shuffle,
              mma = [<MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>],
              subgroup_size_choices = [64], max_load_instruction_bits = 128,
              max_workgroup_sizes = [1024, 1024, 1024],
              max_thread_count_per_workgroup = 1024,
              max_workgroup_memory_bytes = 65536,
              max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>,
                  translation_info = #iree_codegen.translation_info<
                      pipeline = #iree_gpu.pipeline<VectorDistribute>
                      workgroup_size = [256, 1, 1] subgroup_size = 64>} {
    %out_init = tensor.empty() : tensor<2x128x64xf16>
    %res = iree_linalg_ext.attention {
        root_op = #iree_codegen.root_op<set = 0>,
        indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> ()>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
        ins(%q, %k, %v, %scale : tensor<2x128x64xf16>, tensor<2x128x64xf16>,
                                  tensor<2x128x64xf16>, f16)
        outs(%out_init : tensor<2x128x64xf16>) {
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score : f32
         } -> tensor<2x128x64xf16>
    return %res : tensor<2x128x64xf16>
  }
}
"""

# Path to the python-side parity snapshot. Generated on branch
# attn-parity-old-snapshot at commit abb556619a via
# `model_tuner/parity_check.py --side=python`. See
# `model_tuner/attn_parity_results.txt` for the per-MMA-pair breakdown.
_PYTHON_SNAPSHOT = (
    Path(__file__).parent.parent / "model_tuner" / "attn_parity_python.txt"
)


def _extract_signature(line: str) -> tuple[str, str, str, str]:
    """Pull the semantic tuning signature (qk_mma, pv_mma,
    workgroup_tile, reduction_tile) out of a parity-check output line.

    Raises if any field is missing — a silent "?" placeholder would let
    both python and compiler sides converge on the same missing-field
    value and falsely report parity. If IREE's attribute pretty-printer
    changes, that change MUST surface here as a hard failure, not as a
    silent test pass.
    """
    qk = re.search(r"qk_attrs = \{[^}]*mma_kind = #iree_gpu\.(\w+)<([^,>]+)", line)
    pv = re.search(r"pv_attrs = \{[^}]*mma_kind = #iree_gpu\.(\w+)<([^,>]+)", line)
    wg = re.search(r"workgroup = \[([^\]]+)\]", line)
    rd = re.search(r"reduction = \[([^\]]+)\]", line)
    if not (qk and pv and wg and rd):
        missing = [
            name
            for name, match in [
                ("qk_mma", qk),
                ("pv_mma", pv),
                ("workgroup", wg),
                ("reduction", rd),
            ]
            if not match
        ]
        raise ValueError(
            f"failed to extract signature field(s) {missing} from line; "
            f"IREE attr pretty-print format likely changed. Line preview: "
            f"{line[:200]!r}..."
        )
    return (
        f"{qk.group(1)}<{qk.group(2).strip()}>",
        f"{pv.group(1)}<{pv.group(2).strip()}>",
        wg.group(1).strip(),
        rd.group(1).strip(),
    )


def _load_python_signatures() -> set[tuple[str, str, str, str]]:
    """Load semantic signatures from the python-side parity snapshot."""
    assert _PYTHON_SNAPSHOT.is_file(), (
        f"missing python parity snapshot at {_PYTHON_SNAPSHOT}. "
        f"Regenerate via `model_tuner/parity_check.py --side=python` "
        f"on the attn-parity-old-snapshot branch."
    )
    return {
        _extract_signature(line.strip())
        for line in _PYTHON_SNAPSHOT.read_text().splitlines()
        if line.strip()
    }


def test_attention_compiler_bridge_end_to_end() -> None:
    with common.TunerContext() as tuner_ctx:
        ctx = tuner_ctx.mlir_ctx
        with ctx, ir.Location.unknown():
            module = ir.Module.parse(_FIXTURE_MLIR)
        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1, f"expected 1 root op, got {len(root_ops)}"
        tuner = ROCmAttentionVectorDistributeTuner(root_ops[0], tuner_ctx)

        compilation_infos: list[str] = []
        decomposition_configs: list[str] = []
        for solution in candidate_gen.generate_solutions(
            input_module=module,
            mlir_ctx=ctx,
            codegen_pipeline=iree_gpu.LoweringPipeline.VectorDistribute,
        ):
            configs = tuner.get_tuning_configurations(
                solution.constraints_op, solution.knob_assignments
            )
            assert len(configs) == 2, (
                "ROCmAttentionVectorDistributeTuner should emit exactly "
                "(compilation_info, decomposition_config) pairs"
            )
            names = sorted(tc.name for tc in configs)
            assert names == [
                "compilation_info",
                "decomposition_config",
            ], f"unexpected config names: {names}"
            for tc in configs:
                if tc.name == "compilation_info":
                    compilation_infos.append(str(tc.configuration))
                else:
                    decomposition_configs.append(str(tc.configuration))

    # Pair-up invariant: per-candidate, we get one of each.
    assert len(compilation_infos) == len(decomposition_configs), (
        f"compilation_info ({len(compilation_infos)}) and decomposition_config "
        f"({len(decomposition_configs)}) counts diverged — the tuner is "
        "emitting them out of pair"
    )
    # Each candidate must be unique within its stream — duplicates would
    # mean the blocking-clause enumeration is broken.
    n = len(compilation_infos)
    assert len(set(compilation_infos)) == n, (
        "duplicate compilation_info — the Z3 blocking clauses are not "
        "ruling out repeated knob assignments"
    )

    # Bijective parity: extract semantic signatures from each
    # compilation_info / decomposition_config pair and compare against
    # the legacy Python Z3 generator's snapshot.
    paired_lines = [
        f"ci={ci}\tdc={dc}" for ci, dc in zip(compilation_infos, decomposition_configs)
    ]
    compiler_signatures = {_extract_signature(line) for line in paired_lines}
    python_signatures = _load_python_signatures()

    only_in_python = python_signatures - compiler_signatures
    only_in_compiler = compiler_signatures - python_signatures

    if only_in_python or only_in_compiler:
        # Format mismatch for diagnostics: show counts and a few examples
        # from each side so the regression is locatable.
        sample_py = sorted(only_in_python)[:3]
        sample_cc = sorted(only_in_compiler)[:3]
        raise AssertionError(
            f"Bijective-parity drift vs the python snapshot at "
            f"{_PYTHON_SNAPSHOT.name}:\n"
            f"  python:   {len(python_signatures)} signatures\n"
            f"  compiler: {len(compiler_signatures)} signatures\n"
            f"  intersection: "
            f"{len(python_signatures & compiler_signatures)}\n"
            f"  only-in-python ({len(only_in_python)}): "
            f"{sample_py}{' ...' if len(only_in_python) > 3 else ''}\n"
            f"  only-in-compiler ({len(only_in_compiler)}): "
            f"{sample_cc}{' ...' if len(only_in_compiler) > 3 else ''}\n"
            f"If this drift is intentional (the IREE-side attention "
            f"emitter changed its candidate space on purpose), "
            f"regenerate the snapshot via `model_tuner/parity_check.py "
            f"--side=python` on the attn-parity-old-snapshot branch and "
            f"commit the new attn_parity_python.txt."
        )
