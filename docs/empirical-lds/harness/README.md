# LDS Phase Harness

This repository contains small HIP benchmarks and profiling scripts for probing
AMD Local Data Share (LDS) behavior.

## Requirements

- AMD GPU with HIP support.
- CMake 3.26 or newer.
- A HIP/ROCm toolchain providing `amdclang++` and CMake HIP language support.
- `rocprofv3` for the profiling scripts.

The project works with either a normal ROCm install, such as `/opt/rocm`, or
the local project venv. The helpers discover the venv automatically when it is
present, and can also use `ROCM_PATH`, `ROCM_SDK_ROOT`, `HIP_PATH`,
`/opt/rocm`, and tools on `PATH`.

When using the project venv, activate it before building or profiling:

```bash
source venv/bin/activate
```

With a system ROCm install, no venv is required if ROCm tools are discoverable:

```bash
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
```

## Installation

The recommended project-local setup uses TheRock Python packages. Pick the
device extra that matches the GPU under test. On MI300X, ROCm reports `gfx942`:

```bash
rocminfo | grep -m1 'Name:.*gfx'
```

Create the venv and install ROCm packages:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[libraries,device-gfx942]"
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[devel]"
rocm-sdk init
rocm-sdk test
```

`rocm[devel]` plus `rocm-sdk init` is required for building this project. The
runtime/libraries install alone creates `_rocm_sdk_core`, which has useful
tools but does not provide the full development layout CMake needs. After
initialization, CMake should discover `_rocm_sdk_devel` first.

The scripts and CMake prefer the project venv when present, then
`ROCM_PATH`/`ROCM_SDK_ROOT`/`HIP_PATH`, `/opt/rocm`, and tools on `PATH`.

## Build

Configure and build the current benchmark:

```bash
cmake -S . -B build -DLDS_GPU_ARCH=gfx942
cmake --build build
```

Replace the GPU target with the target reported by
ROCm for the machine being tested.

If auto-discovery fails, pass the ROCm root and CMake HIP compiler
explicitly.

```bash
cmake -S . -B build \
  -DLDS_GPU_ARCH=gfx942 \
  -DROCM_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/amdclang++
```

With `LDS_GPU_ARCH` set, CMake configures cleanly when HIP is unavailable and
skips the HIP benchmark targets. To make missing HIP support a hard error:

```bash
cmake -S . -B build -DLDS_GPU_ARCH=gfx942 -DLDS_REQUIRE_HIP=ON
```

The CMake build uses native HIP language support. New HIP executables should be
added with the helper in `CMakeLists.txt`:

```cmake
add_hip_executable(my_benchmark
  SOURCES
    bench/my_benchmark.hip.cpp
)
```

`--save-temps=obj` is enabled, so the build directory also contains useful
intermediate `.s`, `.o`, `.bc`, `.hipi`, and `.hipfb` files for inspection
under the target directories in `build/CMakeFiles/`.

## Run the Benchmark Directly

The benchmark reads an access width, a lane mask, a bank-count/stride value,
and a thread count from stdin:

```text
<access_width> <thread_mask> <bank_count> <threads>
```

Use one wavefront worth of threads for the current GPU. This MI300X example
runs a 32-bit full-wavefront stride of `32`:

```bash
printf '32 0xffffffffffffffff 32 64\n' | ./build/lds_phase_mask
```

To query the wavefront size used by the profiling scripts:

```bash
./build/lds_phase_mask --print-wavefront-size
```

For bank-count profiling, the scripts use a full lane mask and use the guess
as the per-thread LDS index stride. If the assumed mapping
`bank = (address / 4) mod bank_count` is correct, guesses that are multiples
of the real bank count should show the high-latency LDS behavior. For phase
profiling, the scripts use two-lane or generated masks with the configured
bank-count assumption.

## Benchmark Payload Tuning

The benchmark payload may need tuning for the GPU, compiler, and ROCm build
being tested. The current `lds_phase_mask` source keeps both payload shapes in
the benchmark so they can be compared without rewriting the harness:

- The default x-macro payload creates independent LDS read results, then
  accumulates them after the timed block. This tends to preserve a cleaner LDS
  read burst, but it uses more VGPRs and can make the compiler schedule some
  waits or accumulation back into the timed section.
- The alternate `accumN<N>` payload performs a compile-time-unrolled repeated
  read/reduce inside the timed block. This usually lowers VGPR pressure, but it
  measures a read-plus-reduction sequence and can insert more intermediate
  waits and vector adds.

Neither payload is guaranteed to classify every access width on every machine.
When changing the payload or burst length, rebuild and inspect the generated
assembly in `build/CMakeFiles/lds_phase_mask.dir/bench/` to confirm the timed
region still contains the intended number and kind of `ds_read_*` instructions.

The expected result is not just a printed classification; the verbose latency
rows should show a visible gap between the low- and high-latency buckets.
LDS Conflict Counters are useful diagnostics when available, but benchmark latency is the
classification signal because PMC counters can be unavailable or incoherent on
some ROCm/GPU combinations.

## Profiling Scripts

Use `scripts/profile_bank_count.py` to empirically determine the number of LDS banks using measured latencies.
The script queries the benchmark's wavefront size automatically and passes that
thread count to the benchmark, so the user-facing sweep inputs are access width
and bank-count guesses.

Basic run:

```bash
python3 scripts/profile_bank_count.py
```

Recommended explicit sweep:

```bash
python3 scripts/profile_bank_count.py \
  --access-width 32 \
  --guesses 1 2 4 8 16 32 64 96 128 \
  --runs 3
```

Range sweep:

```bash
python3 scripts/profile_bank_count.py --start 1 --stop 128 --step 1
```

Useful options:

- `--access-width {32,64,128}`: select the LDS access width to test,
  defaulting to `32`.
- `--guesses ...`: explicit guesses to test.
- `--start`, `--stop`, `--step`: generate a guess range.
- `-v`, `--verbose`: print the per-guess counter/timer table.

The benchmark validates whether a selected guess indexes past its LDS array and
reports an error before launching the kernel. The Python scripts only validate
the user-facing sweep shape, such as positive guesses and supported access
widths.

With `-v`, the bank-count table includes:

- `conflicts/workgroup`: best-effort LDS bank-conflict counter value normalized
  by the benchmark's 4096 launched workgroups, blank when the raw counter is
  unavailable or disabled. If the raw counter fails normalization, the table
  still displays the per-workgroup quotient and prints a warning that the value
  may be nonsensical.
- `avg thread latency`: average timer ticks per masked thread.
- `min thread latency`: minimum timer ticks measured for any active
  thread.
- `max thread latency`: maximum timer ticks measured for any active
  thread.

After profiling, the script classifies with average thread latency. The
best-effort `conflicts/workgroup` counter is reported for comparison, but it is
not used for classification by default because LDS bank-conflict counters can
be unavailable, zero, or incoherent on some ROCm/GPU combinations. The script
splits the selected latency metric using the largest gap between adjacent
buckets, prints the high-value bucket, and chooses the smallest guess in that
bucket as `most_likely_bank_count`. The smallest high-value guess matters because
multiples of the true bank count are expected to be high as well.

Use `scripts/profile_phase_groups.py` to test the current phase-group
discovery harness. The script profiles `build/lds_phase_mask` with two-lane
masks at the configured access width and prints grouped member lists for each
inferred phase group.

```bash
python3 scripts/profile_phase_groups.py --access-width 64
```

Useful phase-group options:

- `--access-width {32,64,128}`: select the LDS access width to test.
- `--bank-count N`: bank count assumption for the phase-group test, defaulting
  to `32`.
- `-v`, `--verbose`: print comparison rows and phase-group progress.

The phase-group classifier is still experimental: it assigns the highest
metric comparisons to the current group and does not yet compute a confidence
score for noisy or flat metric distributions. In default mode it prints the
classification metric, final grouped member lists, and any warnings; `-v`
additionally prints each comparison row as it is collected.

Use `scripts/profile_phase_experiment.py` for direct mask-pattern
experiments against `build/lds_phase_mask`. This is a lower-level
companion to `profile_phase_groups.py`: instead of inferring groups from a
fixed pairwise search, it runs named thread-mask generators and prints one
counter/timer row per mask.

```bash
python3 scripts/profile_phase_experiment.py --no-pmc
```

Available mask patterns:

- `lane0-pairs`: default. Masks thread `0` and thread `n` for every
  `n` from `1` through the queried wavefront size minus one.
- `halving`: masks contiguous low-lane ranges by repeatedly halving the
  wavefront, ending with lane `0` and the fully unmasked `none` case.
- `shift-full`: starts with the full wavefront mask, then shifts it left by one
  lane per row, ending with `none`.
- `window-N`: masks every contiguous `N`-lane window, labeled `0-(N-1)`,
  `1-N`, and so on. `N` may be `1` through `10`, or `16`.

Useful mask-experiment options:

- `--access-width {32,64,128}`: select the LDS access width to test,
  defaulting to `64`.
- `--pattern`: choose the mask generator.
- `--bank-count N`: bank count assumption for masked index generation,
  defaulting to `32`.

Shared profiling options:

- `--executable PATH`: benchmark executable, defaulting to
  `build/lds_phase_mask`.
- `--rocprof PATH`: rocprofv3 executable. Defaults to the project venv when
  present, then ROCm/PATH discovery.
- `--output-root DIR`: raw rocprof output directory, defaulting to `rocprof`.
- `--runs N`: run each observation multiple times and print averages.
- `--rocprof-verbose`: print rocprof commands, stdout, and stderr.
- `--no-pmc`: skip the best-effort raw LDS bank-conflict counter collection
  and leave `conflicts/workgroup` blank.

When using TheRock packages, run `rocm-sdk init` before profiling so the ROCm
tools and metrics metadata are available.

## Raw rocprofv3 Usage

For debugging the profiler without the Python harness:

```bash
ROCM_ROOT=${ROCM_ROOT:-venv/lib/python3.12/site-packages/_rocm_sdk_devel}
```

```bash
rocprofv3 --output-format csv \
  --output-directory rocprof/manual \
  --pmc SQC_LDS_BANK_CONFLICT \
  --kernel-include-regex mask_b \
  -- ./build/lds_phase_mask
```

Then type the benchmark input on stdin, or pipe it in:

```bash
printf '32 0xffffffffffffffff 32 64\n' | rocprofv3 --output-format csv \
  --output-directory rocprof/manual_32 \
  --pmc SQC_LDS_BANK_CONFLICT \
  --kernel-include-regex mask_b \
  -- ./build/lds_phase_mask
```

## Notes

- The current bank-count path classifies using benchmark latency values. Raw
  LDS bank-conflict PMC values are displayed when available, but they are not
  the default classifier.
