#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

#define HIP_CHECK(expression)                                                  \
  do {                                                                         \
    const hipError_t status = expression;                                      \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

#define VARS(X)                                                                \
  X(v1)                                                                        \
  X(v2)                                                                        \
  X(v3)                                                                        \
  X(v4)                                                                        \
  X(v5)                                                                        \
  X(v6)                                                                        \
  X(v7)                                                                        \
  X(v8)                                                                        \
  X(v9)                                                                        \
  X(v10)                                                                       \
  X(v11)                                                                       \
  X(v12)                                                                       \
  X(v13)                                                                       \
  X(v14)                                                                       \
  X(v15)                                                                       \
  X(v16)                                                                       \
  X(v17)                                                                       \
  X(v18)                                                                       \
  X(v19)                                                                       \
  X(v20)                                                                       \
  X(v21)                                                                       \
  X(v22)                                                                       \
  X(v23)                                                                       \
  X(v24)                                                                       \
  X(v25)                                                                       \
  X(v26)                                                                       \
  X(v27)                                                                       \
  X(v28)                                                                       \
  X(v29)                                                                       \
  X(v30)                                                                       \
  X(v31)                                                                       \
  X(v32)                                                                       \
  X(v33)                                                                       \
  X(v34)                                                                       \
  X(v35)                                                                       \
  X(v36)                                                                       \
  X(v37)                                                                       \
  X(v38)                                                                       \
  X(v39)                                                                       \
  X(v40)                                                                       \
  X(v41)                                                                       \
  X(v42)                                                                       \
  X(v43)                                                                       \
  X(v44)                                                                       \
  X(v45)                                                                       \
  X(v46)                                                                       \
  X(v47)                                                                       \
  X(v48)                                                                       \
  X(v49)                                                                       \
  X(v50)                                                                       \
  X(v51)                                                                       \
  X(v52)                                                                       \
  X(v53)                                                                       \
  X(v54)                                                                       \
  X(v55)                                                                       \
  X(v56)                                                                       \
  X(v57)                                                                       \
  X(v58)                                                                       \
  X(v59)                                                                       \
  X(v60)                                                                       \
  X(v61)                                                                       \
  X(v62)                                                                       \
  X(v63)                                                                       \
  X(v64)

// Default payload: generate distinct ds_reads and accumulate them after timing.
// The memory barrier prevents the compiler from eliminating repeated loads.
// This usually produces a cleaner read burst than accumN, but has high VGPR
// pressure. For wider accesses, the compiler may still move some waits and
// accumulation into the timed section to reduce register pressure.
#define ASSIGN(X)                                                              \
  T X = arr[idx];                                                              \
  asm volatile("" ::: "memory");

#define ACCUM(X) accum += X;

constexpr int MAX_THREADS = 64;
constexpr int MAX_BANK_COUNT = 128;
constexpr int LDS_BANK_BYTES = 4;
constexpr int MAX_LDS_DWORD_INDEX = (MAX_THREADS - 1) * MAX_BANK_COUNT;
constexpr int BLOCKS = 4096;

// clock64, which is the cycle count, is a significantly clearer signal when
// available
__device__ unsigned long long read_timer() {
// HIP docs state clock64 does not work on gfx11 graphics processors
#if defined(__GFX11__)
  return wall_clock64();
#else
  return clock64();
#endif
}

// Alternate tuning payload: compile-time generate a repeated LDS read/reduce.
// This can replace VARS(ASSIGN) while tuning a machine. It produces more
// intermediate waits and vector adds, but has lower register pressure.
// It can improve b128 separation while weakening b32/b64 on some machines.
template <int N, typename T> __device__ T accumN(T *arr, const int idx) {
  T acc{};
#pragma unroll
  for (int i = 0; i < N; ++i) {
    acc += arr[idx];
    asm volatile("" ::: "memory");
  }
  return acc;
}

// Force threads in the thread_mask to all access the same bank
template <typename T>
__global__ void mask_b(T *out, unsigned long long *elapsed, T init,
                       const unsigned long long thread_mask,
                       const int bank_count) {
  constexpr int element_count =
      MAX_LDS_DWORD_INDEX / (sizeof(T) / LDS_BANK_BYTES) + 1;
  __shared__ T arr[element_count];
  for (int j = threadIdx.x; j < element_count; j += blockDim.x) {
    arr[j] = init;
  }
  __syncthreads();

  T accum{};
  const bool masked =
      (thread_mask & (1ULL << static_cast<unsigned int>(threadIdx.x))) != 0;
  // Number of elements before the same bank is accessed again
  const int offset = bank_count / (sizeof(T) / 4);
  const int idx = threadIdx.x * offset;

  if (masked) {
    __syncthreads();
    const unsigned long long start = read_timer();
    // Perform a burst of ds_reads
    VARS(ASSIGN)
    // accum = accumN<64>(arr, idx);
    // accum = accumN<32>(arr, idx);
    __syncthreads();
    const unsigned long long stop = read_timer();
    elapsed[threadIdx.x + blockIdx.x * blockDim.x] = stop - start;
    // Use reads outside of timed block
    VARS(ACCUM)
  }
  out[threadIdx.x + blockIdx.x * blockDim.x] = accum;
}

// uint4, uint2 are not printable. Return a printable value derived from all
// indices
template <typename T> int print_val(T in) {
  if constexpr (std::is_same_v<T, uint4>) {
    return in[0] + in[1] + in[2] + in[3];
  } else if constexpr (std::is_same_v<T, uint2>) {
    return in[0] + in[1];
  } else {
    return in;
  }
}

template <typename T>
int dispatch_kernel(unsigned long long mask, int bank_count, int threads,
                    T init) {
  constexpr int element_count =
      MAX_LDS_DWORD_INDEX / (sizeof(T) / LDS_BANK_BYTES) + 1;
  T *d_out = nullptr;
  T *h_out = nullptr;
  unsigned long long *d_elapsed = nullptr;
  unsigned long long *h_elapsed = nullptr;
  int max_index = 0;
  int masked_threads = 0;
  for (int thread = 0; thread < threads; ++thread) {
    const bool masked = (mask & (1ULL << thread)) != 0;
    if (!masked) {
      continue;
    }
    ++masked_threads;
    const int index = thread * (bank_count / (sizeof(T) / 4));
    if (index > max_index) {
      max_index = index;
    }
  }
  if (max_index >= element_count) {
    std::cerr << "bank count " << bank_count
              << " indexes past the LDS array. Max index is " << max_index
              << "\n";
    return 1;
  }
  const size_t out_size = static_cast<size_t>(threads) * BLOCKS * sizeof(T);
  const size_t elapsed_size =
      static_cast<size_t>(threads) * BLOCKS * sizeof(unsigned long long);

  HIP_CHECK(hipMalloc(&d_out, out_size));
  HIP_CHECK(hipHostMalloc(&h_out, out_size));
  HIP_CHECK(hipMalloc(&d_elapsed, elapsed_size));
  HIP_CHECK(hipHostMalloc(&h_elapsed, elapsed_size));
  mask_b<T><<<BLOCKS, threads>>>(d_out, d_elapsed, init, mask, bank_count);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out, d_out, out_size, hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(h_elapsed, d_elapsed, elapsed_size, hipMemcpyDeviceToHost));

  unsigned long long min_thread_cycles =
      std::numeric_limits<unsigned long long>::max();
  unsigned long long max_thread_cycles = 0;
  unsigned long long total_workgroup_cycles =
      0; // Sum of longest-running thread times
  unsigned long long total_active_thread_cycles = 0;
  for (int block = 0; block < BLOCKS; ++block) {
    unsigned long long block_cycles = 0;
    for (int thread = 0; thread < threads; ++thread) {
      // Unmasked threads should be ignored
      if (!(mask & (1ULL << thread))) {
        continue;
      }
      const unsigned long long thread_cycles =
          h_elapsed[thread + block * threads];
      block_cycles = std::max(block_cycles, thread_cycles);
      min_thread_cycles = std::min(min_thread_cycles, thread_cycles);
      max_thread_cycles = std::max(max_thread_cycles, thread_cycles);
      total_active_thread_cycles += thread_cycles;
    }
    total_workgroup_cycles += block_cycles;
  }

  const double wg_cycles =
      static_cast<double>(total_workgroup_cycles) / static_cast<double>(BLOCKS);
  const double active_thread_cycles =
      masked_threads == 0 ? 0.0
                          : static_cast<double>(total_active_thread_cycles) /
                                static_cast<double>(BLOCKS * masked_threads);
  const unsigned long long displayed_min_thread_cycles =
      masked_threads == 0 ? 0 : min_thread_cycles;
  std::cout << std::fixed << std::setprecision(3)
            << "cycles/workgroup=" << wg_cycles << "\n"
            << "cycles/active thread=" << active_thread_cycles << "\n"
            << "min thread latency=" << displayed_min_thread_cycles << "\n"
            << "max thread latency=" << max_thread_cycles << "\n"
            << "blocks=" << BLOCKS << "\n";

  for (int i = 0; i < threads; ++i) {
    std::cout << "Thread " << i << " calculated " << print_val<T>(h_out[i])
              << "\n";
  }
  HIP_CHECK(hipFreeHost(h_elapsed));
  HIP_CHECK(hipFree(d_elapsed));
  HIP_CHECK(hipFreeHost(h_out));
  HIP_CHECK(hipFree(d_out));
  return 0;
}

__global__ void read_wavefront_size(int *out) {
  *out = __builtin_amdgcn_wavefrontsize();
}

int get_wavefront_size() {
  int *d_wavefront_size = nullptr;
  int h_wavefront_size = 0;
  HIP_CHECK(hipMalloc(&d_wavefront_size, sizeof(int)));
  read_wavefront_size<<<1, 1>>>(d_wavefront_size);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(&h_wavefront_size, d_wavefront_size, sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_wavefront_size));
  return h_wavefront_size;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--print-wavefront-size") == 0) {
    std::cout << get_wavefront_size() << "\n";
    return 0;
  }

  unsigned long long thread_mask = 0;
  int bank_count = 0;
  int threads = 0;
  int access_width = 0;
  if (!(std::cin >> access_width >> std::setbase(0) >> thread_mask >>
        std::dec >> bank_count >> threads)) {
    std::cerr << "expected input: <access_width> <thread_mask> <bank_count> "
                 "<threads>\n";
    return 1;
  }
  if (threads == 0) {
    std::cerr << "thread count must be positive\n";
    return 1;
  }
  if (threads > 64) {
    std::cerr << "thread count must be at most 64 for a 64-bit thread mask\n";
    return 1;
  }
  if (threads < 64 && (thread_mask >> threads) != 0) {
    std::cerr << "thread mask contains lanes outside [0, " << (threads - 1)
              << "]\n";
    return 1;
  }
  if (bank_count <= 0) {
    std::cerr << "bank count must be positive\n";
    return 1;
  }
  if (access_width != 32 && access_width != 64 && access_width != 128) {
    std::cerr << "access width must be either 32, 64, or 128\n";
    return 1;
  }
  if (access_width == 32) {
    return dispatch_kernel<int>(thread_mask, bank_count, threads, 1);
  } else if (access_width == 64) {
    return dispatch_kernel<uint2>(thread_mask, bank_count, threads, {1, 1});
  } else {
    return dispatch_kernel<uint4>(thread_mask, bank_count, threads,
                                  {1, 1, 1, 1});
  }
}
