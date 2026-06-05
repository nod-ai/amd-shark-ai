# RDNA4 RX 9070 XT LDS Measurements

Measurements were collected on AMD Radeon RX 9070 XT (`gfx1201`), which has a
wavefront size of 32 threads. The LDS bank count was measured to be 32. With
4-byte banks, the bank mapping repeats every 128 bytes, so an access stride of
128 bytes causes the maximum possible number of bank conflicts.

Rocprof bank-conflict counters reported zero for every profiled trial on
`gfx1201`, so the tables below use benchmark timer values instead of PMC
conflict counts. This is a known issue for `gfx1201`:
<https://github.com/ROCm/rocm-systems/issues/5953>.

Each table starts at the natural byte stride for that access width: 4 bytes for
`ds_read_b32`, 8 bytes for `ds_read_b64`, and 16 bytes for `ds_read_b128`.
Latencies are average active-thread timer values from `clock64()`, reported in
cycles, and were collected using the macro payload with 64 `ds_read`
instructions.

## ds_read_b32 Latencies

| stride (bytes) | avg thread latency (cycles) |
| -------------- | --------------------------- |
| 4              | 692.318                     |
| 8              | 708.720                     |
| 16             | 840.545                     |
| 32             | 977.425                     |
| 64             | 1531.946                    |
| 128            | 2896.937                    |
| 256            | 2927.956                    |
| 512            | 2904.402                    |

When performing a `ds_read_b32`, lanes access LDS in a single phase: T0-T31.
The high-latency bucket starts at the 128-byte stride and remains saturated for
256- and 512-byte strides.

## ds_read_b64 Latencies

| stride (bytes) | avg thread latency (cycles) |
| -------------- | --------------------------- |
| 8              | 787.120                     |
| 16             | 888.074                     |
| 32             | 1045.397                    |
| 64             | 1551.998                    |
| 128            | 3096.335                    |
| 256            | 3088.239                    |
| 512            | 3080.073                    |

For `ds_read_b64`, the access happens in two phases of 16 lanes each: T0-T15,
then T16-T31. This was observed by comparing pairwise latencies of threads
mapping to the same bank. Threads in the same access phase experience bank
conflicts.

## ds_read_b128 Latencies

| stride (bytes) | avg thread latency (cycles) |
| -------------- | --------------------------- |
| 16             | 1021.298                    |
| 32             | 1139.164                    |
| 64             | 1675.162                    |
| 128            | 3441.533                    |
| 256            | 3495.983                    |
| 512            | 3498.177                    |

For `ds_read_b128`, the access happens in four phases of 8 lanes each: T0-T7,
T8-T15, T16-T23, then T24-T31. Measured using the same method described above.

## Summary

On RX 9070 XT, these measurements support a 32-bank, 4-byte bank model: the
latency pattern saturates at 128-byte stride, and larger strides stay in the
same high-latency bucket.

| access width | phase count | lanes per phase | most likely bank count |
| ------------ | ----------- | --------------- | ---------------------- |
| 32-bit       | 1           | 32              | 32                     |
| 64-bit       | 2           | 16              | 32                     |
| 128-bit      | 4           | 8               | 32                     |

The profiler classified each sweep using average thread latency because the
bank-conflict PMC counter remained zero. Average thread latency shows the same
saturated high-latency region once the stride maps each participating lane in a
hardware LDS phase back to the same bank group.
