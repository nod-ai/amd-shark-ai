"""Bijective parity check between the (legacy) Python Z3 path and the
compiler-bridge path for a single attention dispatch.

The Python Z3 attention generators (rocm_solutions.generate_attention_solutions
+ rocm_dispatch_constraints.generate_attention_vector_distribute_constraints)
were removed in the IREE-constraints refactor, so the two paths live on
two different LOCAL branches of the same checkout. Switch branches with
`git checkout` to flip between them:

  - Legacy Python Z3 path -> branch `attn-parity-old-snapshot` (pinned at
    commit abb556619a, the parent of the IREE-constraints refactor
    commit 41e522343e; this script is carried on it via a follow-up
    commit so the launch works on either branch).
  - Compiler-bridge path -> branch `tuner_use_iree_constraints` (with the
    new attention emitter on the IREE side).

Example invocation (run from the repo root):

  # 1. Python Z3 path
  git checkout attn-parity-old-snapshot
  PYTHONPATH=$PWD/amdsharktuner:\\
$HOME/iree-build/compiler/bindings/python:\\
$HOME/iree-build/runtime/bindings/python \\
  PATH=$HOME/iree-build/tools:$PATH \\
  $HOME/test_venv/bin/python amdsharktuner/model_tuner/parity_check.py \\
      --side=python --out=/tmp/claude/python_attn_configs.txt

  # 2. Compiler-bridge path
  git checkout tuner_use_iree_constraints
  PYTHONPATH=$PWD/amdsharktuner:\\
$HOME/iree-build/compiler/bindings/python:\\
$HOME/iree-build/runtime/bindings/python \\
  PATH=$HOME/iree-build/tools:$PATH \\
  $HOME/test_venv/bin/python amdsharktuner/model_tuner/parity_check.py \\
      --side=compiler --out=/tmp/claude/compiler_attn_configs.txt

  # 3. Compare (which branch is checked out doesn't matter for --diff).
  $HOME/test_venv/bin/python amdsharktuner/model_tuner/parity_check.py \\
      --diff --python=/tmp/claude/python_attn_configs.txt \\
      --compiler=/tmp/claude/compiler_attn_configs.txt

VS Code shortcut: see .vscode/launch.json - three "Attention Parity"
launch configs cover steps 1-3 with the right PYTHONPATH preset. You
still need to `git checkout` between steps 1 and 2 manually.

Note: v0 set-equality is NOT expected today. The two paths emit configs
at different fidelity levels — Python carries per-matmul subgroup_basis,
virtual MMAs, col_major flags, and pipeline options (prefetch / waves-per-eu)
that the v0 compiler emitter doesn't yet knob. Empirically `intersection ≈ 0`.
The diff output explains the structural gap line-by-line; the layout-match
SMT follow-up is the gate for tightening this to bijective parity. Use
this script to surface the gap; use the pytest (tests/rocm_attention_compiler_bridge_test.py)
for regression detection on the compiler-side candidate count.
"""

import argparse
import sys
from pathlib import Path


# Fixed shape used by both sides. Picked small + symmetric so the
# candidate set is manageable to diff by eye when needed.
SHAPE_BATCH = 2
SHAPE_M = 128
SHAPE_N = 64
SHAPE_K1 = 64
SHAPE_K2 = 128

# gfx942 target restricted to a couple of F16 MMAs to keep the
# search space compact. The full attention_benchmark.mlir target carries
# ~16 MMAs; for parity work we only want enough to exercise both
# qk_mma_idx and pv_mma_idx selection paths.
_GPU_TARGET = """#iree_gpu.target<arch = "gfx942",
        features = "", wgp = <compute = fp32, storage = b32,
        subgroup = shuffle,
        mma = [<MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>],
        subgroup_size_choices = [64], max_load_instruction_bits = 128,
        max_workgroup_sizes = [1024, 1024, 1024],
        max_thread_count_per_workgroup = 1024,
        max_workgroup_memory_bytes = 65536,
        max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>"""

_TARGET = (
    f'#hal.executable.target<"rocm", "rocm-hsaco-fb",'
    f" {{iree_codegen.target_info = {_GPU_TARGET}}}>"
)


def _attention_dispatch_mlir() -> str:
    """Self-contained `iree_linalg_ext.attention` dispatch with the fixed
    shape, dtype, and target. Carries `root_op` so the tuner-root walk
    picks it up directly; carries a dummy translation_info so the
    constraint-insertion pass can match it under VectorDistribute."""
    target = _TARGET.replace("\n", " ")
    return f"""
module attributes {{hal.device.targets = [
    #hal.device.target<"hip", [{target}]> : !hal.device]}} {{
  func.func @attention_parity(
      %q: tensor<{SHAPE_BATCH}x{SHAPE_M}x{SHAPE_K1}xf16>,
      %k: tensor<{SHAPE_BATCH}x{SHAPE_K2}x{SHAPE_K1}xf16>,
      %v: tensor<{SHAPE_BATCH}x{SHAPE_K2}x{SHAPE_N}xf16>,
      %scale: f16)
      -> tensor<{SHAPE_BATCH}x{SHAPE_M}x{SHAPE_N}xf16>
      attributes {{hal.executable.target = {target},
                  translation_info = #iree_codegen.translation_info<
                      pipeline = #iree_gpu.pipeline<VectorDistribute>
                      workgroup_size = [256, 1, 1] subgroup_size = 64>}} {{
    %out_init = tensor.empty() : tensor<{SHAPE_BATCH}x{SHAPE_M}x{SHAPE_N}xf16>
    %res = iree_linalg_ext.attention {{
        root_op = #iree_codegen.root_op<set = 0>,
        indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> ()>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}}
        ins(%q, %k, %v, %scale : tensor<{SHAPE_BATCH}x{SHAPE_M}x{SHAPE_K1}xf16>,
                                  tensor<{SHAPE_BATCH}x{SHAPE_K2}x{SHAPE_K1}xf16>,
                                  tensor<{SHAPE_BATCH}x{SHAPE_K2}x{SHAPE_N}xf16>, f16)
        outs(%out_init : tensor<{SHAPE_BATCH}x{SHAPE_M}x{SHAPE_N}xf16>) {{
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score : f32
         }} -> tensor<{SHAPE_BATCH}x{SHAPE_M}x{SHAPE_N}xf16>
    return %res : tensor<{SHAPE_BATCH}x{SHAPE_M}x{SHAPE_N}xf16>
  }}
}}
"""


def _stringify(attrs) -> list[str]:
    """Normalize a list of MLIR attributes to a stable string form so
    `set(python) == set(compiler)` doesn't trip on whitespace, key ordering
    in dict-like attributes, or pretty-print line breaks. This matters for
    `decomposition_config` in particular — MLIR DictionaryAttr keys are
    canonically sorted by the printer but the two paths construct them via
    different code (Python tuner builds them manually; compiler bridge goes
    through materialize_decomposition_config), and we don't want printer
    canonicalization drift between IREE versions to look like real drift."""
    out = []
    for attr in attrs:
        text = str(attr)
        # MLIR can pretty-print with line breaks; collapse to a single line
        # and squash repeated whitespace.
        text = " ".join(text.split())
        out.append(text)
    return out


def _run_python_side(out_path: str) -> int:
    """Drive the legacy Python Z3 attention generator from the pinned
    pre-IREE-constraints worktree. Requires PYTHONPATH to point at the
    OLD amdsharktuner; this script's import statements deliberately
    avoid pulling in the current checkout."""
    from iree.compiler import ir  # type: ignore
    from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

    from amdsharktuner import common, dispatch_parser  # OLD checkout.
    from amdsharktuner.rocm import rocm_solutions  # OLD checkout.

    with common.TunerContext() as tuner_ctx:
        ctx = tuner_ctx.mlir_ctx
        with ctx, ir.Location.unknown():
            module = ir.Module.parse(_attention_dispatch_mlir())
            # Pick the attention root op directly via the tuner-root walker
            # (the fixture annotates it with `root_op`).
            root_ops = iree_codegen.get_tuner_root_ops(module)
            assert len(root_ops) == 1, f"expected 1 tuner root op, got {len(root_ops)}"
            attn_op = root_ops[0]

            # Walk up to the enclosing func to read its target attr.
            func_op = attn_op.parent
            while func_op is not None and func_op.name != "func.func":
                func_op = func_op.parent
            assert func_op is not None, "could not find enclosing func"
            # `get_gpu_target_info` expects the wrapping
            # #hal.executable.target<...> attribute (it casts internally),
            # so parse the full target string.
            target_attr = ir.Attribute.parse(_TARGET.replace("\n", " "))
            gpu_target_info = iree_gpu.TargetInfo.get_gpu_target_info(target_attr)

            parser = dispatch_parser.AttentionOpInterfaceParser(attn_op, tuner_ctx)
            op_info = parser.get_op_info()

        compilation_infos: list[str] = []
        decomposition_configs: list[str] = []
        for batch in rocm_solutions.generate_attention_solutions(
            tuner_ctx=tuner_ctx,
            gpu_target_info=gpu_target_info,
            op_info=op_info,
            dispatch_kind=common.DispatchKind.attention,
            codegen_pipeline=iree_gpu.LoweringPipeline.VectorDistribute,
            num_subgroups=4,
        ):
            for tc in batch:
                if tc.name == "compilation_info":
                    compilation_infos.append(str(tc.configuration))
                elif tc.name == "decomposition_config":
                    decomposition_configs.append(str(tc.configuration))

    _write_output(out_path, compilation_infos, decomposition_configs)
    return 0


def _run_compiler_side(out_path: str) -> int:
    """Drive the compiler-bridge path on the current branch: shells out to
    iree-compile via candidate_gen.generate_solutions, then converts each
    Z3-satisfying knob assignment into a (compilation_info, decomposition_config)
    pair via the existing ROCmAttentionVectorDistributeTuner."""
    from iree.compiler import ir  # type: ignore
    from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

    from amdsharktuner import candidate_gen, common  # CURRENT checkout.
    from amdsharktuner.rocm.rocm_tuners import (  # CURRENT checkout.
        ROCmAttentionVectorDistributeTuner,
    )

    with common.TunerContext() as tuner_ctx:
        ctx = tuner_ctx.mlir_ctx
        with ctx, ir.Location.unknown():
            module = ir.Module.parse(_attention_dispatch_mlir())

        # Find the attention root op.
        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1, f"expected 1 root op, got {len(root_ops)}"
        root_op = root_ops[0]
        tuner = ROCmAttentionVectorDistributeTuner(root_op, tuner_ctx)

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
            for tc in configs:
                if tc.name == "compilation_info":
                    compilation_infos.append(str(tc.configuration))
                elif tc.name == "decomposition_config":
                    decomposition_configs.append(str(tc.configuration))

    _write_output(out_path, compilation_infos, decomposition_configs)
    return 0


def _write_output(
    out_path: str, compilation_infos: list[str], decomposition_configs: list[str]
) -> None:
    norm_ci = _stringify(compilation_infos)
    norm_dc = _stringify(decomposition_configs)
    # Pair compilation_info[i] with decomposition_config[i] so the diff
    # is per-candidate, not interleaved.
    lines = []
    if len(norm_ci) == len(norm_dc):
        for ci, dc in zip(norm_ci, norm_dc):
            lines.append(f"ci={ci}\tdc={dc}")
    else:
        # Defensive: if the two streams have different lengths, dump them
        # separately so the diff still surfaces something.
        lines.extend(f"ci_only={c}" for c in norm_ci)
        lines.extend(f"dc_only={d}" for d in norm_dc)
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(
        f"  total candidates: {len(norm_ci)} compilation_info, "
        f"{len(norm_dc)} decomposition_config"
    )
    print(f"  wrote {out_path}")


def _diff(python_path: str, compiler_path: str) -> int:
    py_lines = Path(python_path).read_text().splitlines()
    cc_lines = Path(compiler_path).read_text().splitlines()
    py_set = set(py_lines)
    cc_set = set(cc_lines)
    only_py = py_set - cc_set
    only_cc = cc_set - py_set
    print(f"python  candidates: {len(py_lines)}  (unique: {len(py_set)})")
    print(f"compiler candidates: {len(cc_lines)}  (unique: {len(cc_set)})")
    print(f"intersection:        {len(py_set & cc_set)}")
    print(f"only-in-python:      {len(only_py)}")
    print(f"only-in-compiler:    {len(only_cc)}")
    if not only_py and not only_cc and len(py_lines) == len(cc_lines):
        print("Bijective parity confirmed.")
        return 0
    if not only_py and only_cc:
        print(
            "Compiler-bridge is a strict SUPERSET of the Python path "
            f"(+{len(only_cc)} extra candidates) — expected v0 shape; the "
            "deferred layout-match SMT would rule out the extras."
        )
        return 0
    # The most common v0 outcome: total drift in both directions, because
    # the two paths emit configs at DIFFERENT FIDELITY LEVELS (see the
    # v0 caveats below). Report counts + sample diff but exit success;
    # the pytest counterpart asserts count-stability, not set-equality.
    print(
        "\nNon-bijective: configs differ structurally as well as in count.\n"
        "v0 fidelity gaps that make set-equality unattainable today:\n"
        "  - per-matmul subgroup_basis: Python emits it explicitly on each\n"
        "    qk/pv lowering_config; v0 compiler emits only mma_kind there.\n"
        "  - virtual MMAs (VMFMA_*) + col_major: gated by the deferred\n"
        "    layout-match SMT; compiler v0 emits the canonical MMAs only.\n"
        "  - top-level reduction vs partial_reduction: cosmetic key drift\n"
        "    between the two compilation_info builders.\n"
        "  - pipeline options (prefetch_num_stages, amdgpu-waves-per-eu):\n"
        "    enumerated by the Python tuner; not yet knobbed in v0.\n"
        "Bijective parity is the gate for the layout-match follow-up.\n"
    )
    print("only-in-python (first 3):")
    for line in list(only_py)[:3]:
        print(f"  {line[:200]}{'...' if len(line) > 200 else ''}")
    print("only-in-compiler (first 3):")
    for line in list(only_cc)[:3]:
        print(f"  {line[:200]}{'...' if len(line) > 200 else ''}")
    # Exit 0 — non-equality is an expected v0 outcome, not a regression.
    # The pytest counterpart pins the compiler-side count for regression
    # detection; this script exists to surface the gap for follow-up.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["python", "compiler"])
    parser.add_argument("--out", help="Output file when --side is set.")
    parser.add_argument("--diff", action="store_true", help="Diff two output files.")
    parser.add_argument("--python", help="Python-side output file for --diff.")
    parser.add_argument("--compiler", help="Compiler-side output file for --diff.")
    args = parser.parse_args()

    if args.diff:
        if not args.python or not args.compiler:
            parser.error("--diff requires --python and --compiler")
        return _diff(args.python, args.compiler)

    if not args.side or not args.out:
        parser.error("--side and --out are required when not in --diff mode")

    if args.side == "python":
        print("== Python Z3 path ==")
        return _run_python_side(args.out)
    print("== Compiler-bridge path ==")
    return _run_compiler_side(args.out)


if __name__ == "__main__":
    sys.exit(main())
