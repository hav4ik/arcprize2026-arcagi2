// Minimal zero-copy host all-reduce (the idea of Manh's VLLM_HOST_ALLREDUCE,
// without packing, SIMD intrinsics or prefetch). One kernel per call:
//   writer blocks store this rank's input unit by unit straight into its slot
//   of a shared pinned host segment (mapped, so plain GPU stores cross PCIe),
//   then raise a per-unit flag; CPU reducer threads (one set per rank) spin on
//   the flags, sum the world's slots in fp32 in rank order, round once to bf16
//   and raise a per-unit done flag; reader blocks spin on the done flags and
//   copy each reduced unit back. Writes, CPU sums and reads overlap by unit.
// No cudaMemcpy, no host synchronisation, and the sequence number lives in
// device memory, so the call can be captured in a CUDA graph.
//
// Segment layout (bytes): [0,64) header (seq<<32 | nbytes, posted by rank 0),
// in flags [world][max_units] and done flags [max_units] at 64-byte stride,
// then slots [world][max_bytes], then the result [max_bytes].
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>
#include <time.h>

namespace {

constexpr int kStride = 16;  // uint32 words per flag (64 bytes)

struct Layout {
  int64_t max_bytes, unit, max_units, world;
  int64_t off_in, off_done, off_slots, off_res, total;
};

Layout make_layout(int64_t max_bytes, int64_t unit, int64_t world) {
  Layout l;
  l.max_bytes = max_bytes;
  l.unit = unit;
  l.world = world;
  l.max_units = (max_bytes + unit - 1) / unit;
  l.off_in = 64;
  l.off_done = l.off_in + world * l.max_units * 64;
  l.off_slots = (l.off_done + l.max_units * 64 + 4095) / 4096 * 4096;
  l.off_res = l.off_slots + world * max_bytes;
  l.total = l.off_res + max_bytes;
  return l;
}

__device__ __forceinline__ uint32_t ldv(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void stv(uint32_t* p, uint32_t v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void stv64(uint64_t* p, uint64_t v) {
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void har_kernel(const uint4* __restrict__ x, uint4* __restrict__ out,
                           char* seg, int64_t nbytes, int64_t unit,
                           int64_t max_units, int64_t max_bytes, int64_t off_in,
                           int64_t off_done, int64_t off_slots, int64_t off_res,
                           int rank, int writers, uint32_t* ctr) {
  const uint32_t seq = ctr[0] + 1;
  const int64_t nunits = (nbytes + unit - 1) / unit;
  uint32_t* in_flags = reinterpret_cast<uint32_t*>(seg + off_in);
  uint32_t* done = reinterpret_cast<uint32_t*>(seg + off_done);
  if ((int)blockIdx.x < writers) {
    if (rank == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
      stv64(reinterpret_cast<uint64_t*>(seg), (uint64_t)seq << 32 | (uint64_t)nbytes);
      __threadfence_system();
    }
    uint4* slot = reinterpret_cast<uint4*>(seg + off_slots + rank * max_bytes);
    for (int64_t u = blockIdx.x; u < nunits; u += writers) {
      const int64_t beg = u * unit / 16, end = min(nbytes, (u + 1) * unit) / 16;
      for (int64_t i = beg + threadIdx.x; i < end; i += blockDim.x) slot[i] = x[i];
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) stv(in_flags + (rank * max_units + u) * kStride, seq);
    }
  } else {
    const int readers = gridDim.x - writers, rb = blockIdx.x - writers;
    const uint4* res = reinterpret_cast<const uint4*>(seg + off_res);
    for (int64_t u = rb; u < nunits; u += readers) {
      if (threadIdx.x == 0) {
        const uint64_t t0 = gtimer();
        while (ldv(done + u * kStride) != seq) {
          if (gtimer() - t0 > 2000000000ull || ldv(ctr + 2)) { ctr[2] = 1; break; }
        }
        __threadfence_system();
      }
      __syncthreads();
      const int64_t beg = u * unit / 16, end = min(nbytes, (u + 1) * unit) / 16;
      for (int64_t i = beg + threadIdx.x; i < end; i += blockDim.x) out[i] = __ldcv(res + i);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&ctr[1], 1u) == gridDim.x - 1) {
      ctr[1] = 0;
      ctr[0] = seq;
    }
  }
}

inline float bf2f(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

struct Reducers {
  Layout l;
  char* base = nullptr;
  int rank = 0, threads = 1;
  int64_t poll_ns = 0;  // 0: hot spin (warm); >0: sleep this long between polls (cold)
  std::atomic<bool> stop{false};
  std::vector<std::thread> ts;
  std::atomic<uint64_t> busy_ns{0}, calls{0};
};
Reducers* g = nullptr;

inline void relax(int64_t poll_ns) {
  if (poll_ns > 0) {
    timespec t{0, (long)poll_ns};
    nanosleep(&t, nullptr);
  } else {
    __builtin_ia32_pause();
  }
}

void reducer_loop(Reducers* r, int t) {
  const Layout& l = r->l;
  volatile uint64_t* header = reinterpret_cast<volatile uint64_t*>(r->base);
  volatile uint32_t* in_flags = reinterpret_cast<volatile uint32_t*>(r->base + l.off_in);
  volatile uint32_t* done = reinterpret_cast<volatile uint32_t*>(r->base + l.off_done);
  const int64_t me = r->rank * r->threads + t, nred = l.world * r->threads;
  uint32_t seq = 0;
  while (!r->stop.load(std::memory_order_relaxed)) {
    uint64_t h = *header;
    if ((uint32_t)(h >> 32) <= seq) {
      relax(r->poll_ns);
      continue;
    }
    seq = (uint32_t)(h >> 32);
    const int64_t nbytes = (int64_t)(h & 0xffffffffu);
    const int64_t nunits = (nbytes + l.unit - 1) / l.unit;
    for (int64_t u = me; u < nunits; u += nred) {
      for (int w = 0; w < l.world; ++w)
        while (in_flags[(w * l.max_units + u) * kStride] != seq) {
          if (r->stop.load(std::memory_order_relaxed)) return;
          relax(r->poll_ns);
        }
      std::atomic_thread_fence(std::memory_order_acquire);
      const int64_t beg = u * l.unit / 2, end = std::min(nbytes, (u + 1) * l.unit) / 2;
      const uint16_t* s0 = reinterpret_cast<const uint16_t*>(r->base + l.off_slots);
      uint16_t* res = reinterpret_cast<uint16_t*>(r->base + l.off_res);
      const int64_t stride = l.max_bytes / 2;
      if (l.world == 2) {
        const uint16_t* a = s0;
        const uint16_t* b = s0 + stride;
#pragma GCC ivdep
        for (int64_t i = beg; i < end; ++i) res[i] = f2bf(bf2f(a[i]) + bf2f(b[i]));
      } else {
        for (int64_t i = beg; i < end; ++i) {
          float s = bf2f(s0[i]);
          for (int w = 1; w < l.world; ++w) s += bf2f(s0[w * stride + i]);
          res[i] = f2bf(s);
        }
      }
      std::atomic_thread_fence(std::memory_order_release);
      done[u * kStride] = seq;
    }
  }
}

// Registers [addr, addr + bytes) (a mapped /dev/shm file) as pinned + mapped
// memory and returns its device address.
int64_t register_host(int64_t addr, int64_t bytes) {
  void* p = reinterpret_cast<void*>(addr);
  TORCH_CHECK(cudaHostRegister(p, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess,
              "cudaHostRegister failed");
  void* d = nullptr;
  TORCH_CHECK(cudaHostGetDevicePointer(&d, p, 0) == cudaSuccess, "cudaHostGetDevicePointer failed");
  return reinterpret_cast<int64_t>(d);
}

std::vector<int64_t> layout(int64_t max_bytes, int64_t unit, int64_t world) {
  Layout l = make_layout(max_bytes, unit, world);
  return {l.max_units, l.off_in, l.off_done, l.off_slots, l.off_res, l.total};
}

void start(int64_t addr, int64_t max_bytes, int64_t unit, int64_t world, int64_t rank,
           int64_t threads, int64_t poll_ns) {
  TORCH_CHECK(g == nullptr, "already started");
  g = new Reducers();
  g->l = make_layout(max_bytes, unit, world);
  g->base = reinterpret_cast<char*>(addr);
  g->rank = rank;
  g->threads = threads;
  g->poll_ns = poll_ns;
  for (int t = 0; t < threads; ++t) g->ts.emplace_back(reducer_loop, g, t);
}

void stop() {
  if (!g) return;
  g->stop = true;
  for (auto& t : g->ts) t.join();
  delete g;
  g = nullptr;
}

void all_reduce(torch::Tensor x, torch::Tensor out, int64_t dev_base, int64_t max_bytes,
                int64_t unit, int64_t world, int64_t rank, int64_t writers, int64_t readers,
                int64_t block, torch::Tensor ctr) {
  Layout l = make_layout(max_bytes, unit, world);
  const int64_t nbytes = x.numel() * x.element_size();
  TORCH_CHECK(nbytes % 16 == 0 && nbytes <= max_bytes);
  const int64_t nunits = (nbytes + unit - 1) / unit;
  const int w = (int)std::min<int64_t>(writers, nunits), r = (int)std::min<int64_t>(readers, nunits);
  auto stream = c10::cuda::getCurrentCUDAStream();
  har_kernel<<<w + r, (int)block, 0, stream>>>(
      reinterpret_cast<const uint4*>(x.data_ptr()), reinterpret_cast<uint4*>(out.data_ptr()),
      reinterpret_cast<char*>(dev_base), nbytes, unit, l.max_units, max_bytes, l.off_in,
      l.off_done, l.off_slots, l.off_res, (int)rank, w, reinterpret_cast<uint32_t*>(ctr.data_ptr()));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_host", &register_host);
  m.def("layout", &layout);
  m.def("start", &start);
  m.def("stop", &stop);
  m.def("all_reduce", &all_reduce);
}
