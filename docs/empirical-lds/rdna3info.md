# RDNA3 W7900 LDS Measurements

Measurements were collected on AMD Radeon W7900 (`gfx1100`), which has a
wavefront size of 32 threads. The LDS bank count was measured to be 32. With
4-byte banks, the bank mapping repeats every 128 bytes, so an access stride of
128 bytes causes the maximum possible number of bank conflicts.

Each table starts at the natural byte stride for that access width: 4 bytes for
`ds_read_b32`, 8 bytes for `ds_read_b64`, and 16 bytes for `ds_read_b128`.
Latencies are reported as ticks because cycle-based `clock64()` is unavailable
for `gfx1100`, and were collected using the macro payload with 64 `ds_read`
instructions.

## ds_read_b32 Latencies

| stride (bytes) | avg thread latency (ticks) |
| -------------- | -------------------------- |
| 4              | 38.542                     |
| 8              | 38.971                     |
| 16             | 41.535                     |
| 32             | 49.516                     |
| 64             | 68.829                     |
| 128            | 119.232                    |
| 256            | 118.953                    |
| 512            | 119.898                    |

When performing a `ds_read_b32`, lanes access LDS in a single phase: T0-T31.
The high-latency bucket starts at the 128-byte stride and remains saturated for
256- and 512-byte strides.

## ds_read_b64 Latencies

| stride (bytes) | avg thread latency (ticks) |
| -------------- | -------------------------- |
| 8              | 45.415                     |
| 16             | 47.146                     |
| 32             | 55.371                     |
| 64             | 74.461                     |
| 128            | 129.932                    |
| 256            | 128.193                    |
| 512            | 129.466                    |

For `ds_read_b64`, the access happens in two phases of 16 lanes each: T0-T15,
then T16-T31. This was observed by comparing pairwise latencies of threads
mapping to the same bank. Threads in the same access phase experience bank
conflicts.

## ds_read_b128 Latencies

| stride (bytes) | avg thread latency (ticks) |
| -------------- | -------------------------- |
| 16             | 58.971                     |
| 32             | 66.187                     |
| 64             | 88.664                     |
| 128            | 162.551                    |
| 256            | 156.674                    |
| 512            | 158.890                    |

For `ds_read_b128`, the access is observed to happen in four noncontiguous
phases of 8 lanes each.

1. T0-T3 and T20-T23
2. T4-T7 and T16-T19
3. T8-T11 and T28-T31
4. T12-T15 and T24-T27

## Summary

On W7900, these measurements support a 32-bank, 4-byte bank model: the latency
pattern saturates at 128-byte stride, and larger strides stay in the same
high-latency bucket.

| access width | phase count | lanes per phase | most likely bank count |
| ------------ | ----------- | --------------- | ---------------------- |
| 32-bit       | 1           | 32              | 32                     |
| 64-bit       | 2           | 16              | 32                     |
| 128-bit      | 4           | 8               | 32                     |

The profiler classified each sweep using average thread latency. The reported
tick values are not directly comparable with cycle counts on GPUs where
cycle-based `clock64()` is available, but the stride-dependent latency buckets
show the same saturated region expected from the 32-bank mapping.
