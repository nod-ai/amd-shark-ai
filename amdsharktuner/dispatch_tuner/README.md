# Getting Started with Dispatch Tuner

Example of tuning a dispatch using `dispatch_tuner`

## Environments
Follow instructions in [`/amdsharktuner/README.md`](../README.md)

## Running the Tuner

### Choose a dispatch to tune
This example uses the simple `dispatch_sample.mlir` file.

### Generate a benchmark file
Use the usual `iree-compile` command for problem dispatch, add
`--iree-hal-dump-executable-files-to=dump --iree-codegen-add-tuner-attributes`,
and get the dispatch benchmark that you want to tune. For example:

```shell
iree-compile dispatch_sample.mlir --iree-hal-target-device=hip \
    --iree-rocm-target=gfx942 --iree-hal-dump-executable-files-to=tmp/dump \
    --iree-codegen-add-tuner-attributes -o /dev/null

cp tmp/dump/module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir tmp/dispatch_sample_benchmark.mlir
```

### Recommended Trial Run
For an initial trial to test the tuning loop, use following command:

```shell
cd amdshark-ai/amdsharktuner
python -m dispatch_tuner dispatch_tuner/dispatch_sample.mlir \
    dispatch_tuner/tmp/dispatch_sample_benchmark.mlir \
    --compile-flags-file=dispatch_tuner/compile_flags.txt \
    --devices=hip://0 --num-candidates=30
```

> Example input format for multiple devices: use a comma-separated list, such as `--devices=hip://0,hip://1`


[!TIP]
Use the `--starter-td-spec` option to pass an existing td spec for the run.
You can use following default td spec: [Default Spec](https://github.com/iree-org/iree/blob/main/compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir).

### Matvec example

This example tunes a matvec (matrix-vector multiply) dispatch using the
`VectorDistribute` reduction pipeline. Matvec operations are compiled by
IREE via `setReductionConfig` (not the MMA contraction path), so the tuner
explores a different set of knobs: `subgroup_size`, `thread_loads`,
`workgroup_size`, and `num_parallel_reductions`.

#### Generate a benchmark file

```shell
iree-compile dispatch_tuner/matvec_sample.mlir --iree-hal-target-device=hip \
    --iree-rocm-target=gfx942 --iree-hal-dump-executable-files-to=tmp/matvec_dump \
    --iree-codegen-add-tuner-attributes -o /dev/null

cp tmp/matvec_dump/module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir tmp/matvec_sample_benchmark.mlir
```

#### Run the tuner

> **Important:** Matvec uses the `VectorDistribute` pipeline, not the default
> `TileAndFuse`. You **must** pass `--codegen-pipeline=llvmgpu_vector_distribute`,
> otherwise the tuner will not find a matching dispatch tuner and produce
> zero candidates.

```shell
cd amdshark-ai/amdsharktuner
python -m dispatch_tuner dispatch_tuner/matvec_sample.mlir \
    tmp/matvec_sample_benchmark.mlir \
    --compile-flags-file=dispatch_tuner/compile_flags.txt \
    --codegen-pipeline=llvmgpu_vector_distribute \
    --devices=hip://0 --num-candidates=30
```

#### How to tell if your dispatch needs `--codegen-pipeline=llvmgpu_vector_distribute`

Check the dumped benchmark file for the pipeline IREE chose:

```shell
grep "VectorDistribute\|TileAndFuse" tmp/matvec_dump/*benchmark*.mlir
```

If the output contains `VectorDistribute`, use `--codegen-pipeline=llvmgpu_vector_distribute`.
If it contains `TileAndFuse`, use the default (no flag needed).

#### Supported matvec shapes

The matvec tuner supports contraction-interface ops where exactly one of the
M or N dimension groups is empty:

- `linalg.matvec` (M present, N empty)
- `linalg.vecmat` (M empty, N present)
- `linalg.batch_matvec` and batched variants
- `linalg.generic` ops that IREE recognizes as matvec-shaped contractions

Element types must be 4, 8, 16, or 32-bit. Static shapes only.

## Algorithm of the dispatch tuner
### Tuning algorithm
1. Generate Candidate specs
2. Compile candidate
3. Benchmark for candidates
    - Baseline benchmark for candidates (now serially over all the given devices)
    - Candidate benchmark (parallel over all the given devices)
    - Second baseline run to check for any regression
    - Return top candidates
