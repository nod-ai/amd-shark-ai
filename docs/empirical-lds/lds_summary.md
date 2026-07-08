# Empirical LDS Measurements

These notes record empirical measurements of AMD Local Data Share (LDS)
latencies and phase groups for `ds_read` instructions. The measurements use
strided LDS accesses to force lanes onto specific LDS banks, making bank
conflict behavior visible in benchmark latency.

## Measured LDS Bank Counts

| Architecture | GPU | target | wavefront size | measured bank count |
| ------------ | --- | ------ | -------------- | ------------------- |
| [CDNA4](cdna4info.md) | MI350X | `gfx950` | 64 | 64 |
| [CDNA3](cdna3info.md) | MI300X | `gfx942` | 64 | 32 |
| [RDNA4](rdna4info.md) | RX 9070 XT | `gfx1201` | 32 | 32 |
| [RDNA3](rdna3info.md) | W7900 | `gfx1100` | 32 | 32 |

## Measured LDS Phase Groups

### CDNA4 / MI350X

#### ds_read_b32

- T0-T63

#### ds_read_b64

- T0-T31
- T32-T63

#### ds_read_b128

- T0-T3, T12-T15, T20-T23, T24-T27
- T32-T35, T44-T47, T52-T55, T56-T59
- T4-T7, T8-T11, T16-T19, T28-T31
- T36-T39, T40-T43, T48-T51, T60-T63

### CDNA3 / MI300X

#### ds_read_b32

- T0-T31
- T32-T63

#### ds_read_b64

- T0-T15
- T16-T31
- T32-T47
- T48-T63

#### ds_read_b128

- T0-T3 and T20-T23
- T32-T35 and T52-T55
- T4-T7 and T16-T19
- T36-T39 and T48-T51
- T8-T11 and T28-T31
- T40-T43 and T60-T63
- T12-T15 and T24-T27
- T44-T47 and T56-T59

### RDNA4 / RX 9070 XT

#### ds_read_b32

- T0-T31

#### ds_read_b64

- T0-T15
- T16-T31

#### ds_read_b128

- T0-T7
- T8-T15
- T16-T23
- T24-T31

### RDNA3 / W7900

#### ds_read_b32

- T0-T31

#### ds_read_b64

- T0-T15
- T16-T31

#### ds_read_b128

- T0-T3 and T20-T23
- T4-T7 and T16-T19
- T8-T11 and T28-T31
- T12-T15 and T24-T27

## Methodology

The source experiments used the `lds_phase_mask` HIP benchmark and the
`profile_bank_count.py` profiling harness from the LDS phase harness
project. The runnable harness is preserved in [harness/](harness/README.md).
The benchmark uses a shared 64-read timing block. Threads are mapped pairwise
to the same bank, then reads are performed to check if a conflict occurs. Only
threads belonging to the same bank can conflict.

The profiler records average active-thread latency from the benchmark timer and
collects raw rocprof LDS bank-conflict counters when available. Latency is the
classification signal to determine presence of conflicts because PMC counters
can be unavailable, zero, or incoherent on some ROCm and GPU combinations. The
smallest stride or guess in the high-latency bucket is treated as the likely
bank-count period.
