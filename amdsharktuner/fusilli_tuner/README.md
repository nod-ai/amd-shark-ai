# Fusilli Tuner

Autotuning tool for generating IREE dispatch tuning specs for operations benchmarked using [Fusilli](https://github.com/iree-org/fusilli), a JIT frontend and graph API for IREE.

**Fusilli Command Format**: This tuner accepts convolution and matmul parameters
in Fusilli benchmark driver format. Example commands:

```
conv -F 1 --bf16 -n 16 -c 64 -H 48 -W 32 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC --spatial_dim 2
matmul -M 1024 -N 1024 -K 1024 --a_type bf16 --b_type bf16 --out_type bf16
```

For detailed explanations of Fusilli command parameters, see the
[Fusilli README](https://github.com/iree-org/fusilli).

---

## Prerequisites

### IREE Setup

Follow instructions in [`/amdsharktuner/README.md`](../README.md) for IREE installation and setup.

### Tuner Setup

Set up PYTHONPATH:

```shell
cd amdsharktuner
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Fusilli Benchmark Driver

Build the Fusilli benchmark driver following the
[Fusilli build instructions](https://github.com/iree-org/fusilli#build-and-test).

You'll specify the path using `--fusilli-driver` when running the tuner.

---

## Running the Tuner

### Choose a kernel to tune

This tuner accepts convolution and matmul parameters in Fusilli benchmark driver format.

### Recommended Trial Run

For an initial trial to test the tuning loop, use the following command:

```shell
cd amdshark-ai/amdsharktuner
python -m fusilli_tuner \
  --fusilli-driver ~/fusilli/build/bin/benchmarks/fusilli_benchmark_driver \
  --commands-file fusilli_tuner/example_commands.txt \
  --output-td-spec tuning_spec.mlir \
  --num-candidates 30 \
  --devices hip://0
```

Alternatively, you can pass the Fusilli command directly:

```shell
python -m fusilli_tuner \
  --fusilli-driver ~/fusilli/build/bin/benchmarks/fusilli_benchmark_driver \
  --output-td-spec tuning_spec.mlir \
  --num-candidates 30 \
  --devices hip://0 \
  conv -F 1 --bf16 -n 1 -c 64 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC --spatial_dim 2
```

> Example input format for multiple devices: use a comma-separated list, such
> as `--devices=hip://0,hip://1`

> [!TIP]
> Use the `--starter-td-spec` option to pass an existing td spec for the run.

---

## Tuning Algorithm

1. Run Fusilli benchmark driver with `--dump` to generate source MLIR
2. Extract dispatch benchmarks with `iree-compile`
3. Generate candidate specs, compile, and benchmark
4. Return top candidates

For details on the tuning algorithm, see
[amdshark Tuner Overview](../README.md#tuning-algorithm).

For Fusilli-specific information, see
[Fusilli Documentation](https://github.com/iree-org/fusilli).
