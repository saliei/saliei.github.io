---
title: "SIMD Programming and Vector Optimizations"
date: "2025-02-03"
author: "Saeid"
description: "
Essentially all modern processors have the capability to apply instructions on a vector 
in one processing unit cycle instead of operating on a single scalar. 
Language designers and compiler developers have been trying hard to leverage these 
hardware capabilities by compiling scalar programs into vector instructions. 
One possible approach is using SIMD (Single Instruction, Multiple Data) intrinsics, 
supported by all modern C/C++ compilers, through SSE (Streaming SIMD Extension), 
AVX (Advanced Vector Extensions) others for x86 architectures, and ARM NEON extensions.
"
---
Essentially all modern processors have the capability to apply instructions on a vector 
in one processing unit cycle instead of operating on a single scalar. 
Language designers and compiler developers have been trying hard to leverage these 
hardware capabilities by compiling scalar programs into vector instructions. 
One possible approach is using SIMD (Single Instruction, Multiple Data) intrinsics, 
supported by all modern C/C++ compilers, through SSE (Streaming SIMD Extension), 
AVX (Advanced Vector Extensions) others for x86 architectures, and ARM NEON extensions. 
Modern CPUs provide dedicated registers for these vector instructions:

- x86: SSE (128-bit) - AVX,AVX2 (256-bit) - AVX-512(512-bit).
	- For example, an AVX 256-bit register can store: 8 x 32-bit floats, 4 x 64-bit doubles, or 8 x 32-bit integers. 
- ARM: NEON (128-bit) - SVE (scalable vector length).
- RISC-V: Vector Extension (RVV, scalable).

Key architectural transitions were: MMX (64-bit) → SSE (128-bit) → AVX (256-bit) → AVX-512 (512-bit) → Scalable Vectors. 
Each architecture implements vector processing differently, but they share common principles:

- Vector registers of fixed or scalable width.
- [Predication](https://en.wikipedia.org/wiki/Predication_(computer_architecture)) support.
- Gather-scatter operations.
- Cross-lane operations.

Modern compilers can automatically vectorize code under certain conditions using level 2 or level 3 optimizations. 
Also, the compiler can be guided with `#pragma omp simd`, `#pragma GCC ivdep` directives. 
An example of a vectorization-friendly snippet:

```cpp
// likely to be auto-vectorizable by the compiler
void auto_vectorized_add(float* __restrict__ a, 
						 float* __restrict__ b, 
						 float* __restrict__ c, 
						 size_t n) {
	#pragma omp simd
	for (size_t i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}
```

To check whether vectorization was applied, we can compile as such:

```bash
# -ftree-vectorize will be turned on automatically by -O3
g++ -O2 -std=c++0x -funroll-loops -ftree-vectorize -ftree-vectorizer-verbose=1 auto_vectorized.cpp -o auto_vectorized.x
```

Official [GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) page 
lists an overview of GCC complete optimization flags and at which level they are turned on.
Conditions under which the compiler is likely to auto-vectorize include:

- Aligned memory access.
- No loop-carried dependencies.
- Straight-line code within the loop.
- Known trip counts.

When auto-vectorization fails or isn't optimal, explicit vectorization using intrinsics provides fine-grained control. 
The API for these intrinsics looks similar to library functions, but unlike these library function calls, 
intrinsics are implemented directly in the compiler. For example, `__m128 _mm_add_ps(__m128 A, __m128 B)` SSE intrinsics 
typically will directly map to a single `addps` instruction, which by the time it takes the CPU to call a library function, 
it can complete tens of these low-level instructions. These intrinsics are provided in `immintrin.h`, the header 
for x86, and `arm_neon.h` header for ARM architecture. 
Also, libraries like Eigen, SIMD Everywhere (SIMDe), and XSIMD provide portable SIMD support instead of the raw builtin intrinsics, 
and as of C++24 there is an [experimental standard library](https://en.cppreference.com/w/cpp/experimental/simd/simd) `std::experimental::simd`.  

The x86 intrinsics follow a consistent naming pattern: `__mm<bit_width>_<operation>_<data_type>`, where `<bit_width>` can be empty (128-bit SSE), 
256 or 512 for AVX or AVX-512 respectively and `<data_type>` can be `ps` (packed single), `pd` (packed double), `epi32` (packed 32-bit integers), etc. 
Examples of the x86 intrinsics:

- `_mm256_load_ps()`: aligned load.
- `_mm256_loadu_ps()`: unaligned load.
- `_mm256_store_ps()`: aligned store.
- `_mm256_stream_ps()`: non-temporal store which bypasses cache.

[Intel's intrinsics guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) lists all the x86 intrinsics.
An example using AVX2 to add two array of float values:

```cpp
#include <iostream>
#include <immintrin.h>

void simd_add(float* a, float* b, float* c, size_t n) {
	// process 8 elements at a time
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);  // load 8 floats
        __m256 vb = _mm256_loadu_ps(&b[i]);  // load 8 floats
        __m256 vc = _mm256_add_ps(va, vb);   // SIMD addition
        _mm256_storeu_ps(&c[i], vc);         // store results
    }
}

int main() {
	// align to 32-byte memory address
    alignas(32) float a[8] = {1,2,3,4,5,6,7,8};
    alignas(32) float b[8] = {8,7,6,5,4,3,2,1};
    alignas(32) float c[8];

    simd_add(a, b, c, 8);

    for (float v: c) std::cout << v << std::endl;
}
```

Using the portable libraries, like XSIMD, we would have:

```cpp
#include <vector>
#include <iostream>
#include <xsimd/xsimd.hpp>

void xsimd_add(const std::vector<float>& a, 
			   const std::vector<float>& b, 
			   std::vector<float>& c) {
	auto n = a.size();
	constexpr std::size_t simd_size = xsimd::batch<float>::size;

	for (std::size_t i = 0; i < n; i += simd_size) {
		xsimd::batch<float> va = xsimd::load_unaligned(&a[i]);
		xsimd::batch<float> vb = xsimd::load_unaligned(&b[i]);
		xsimd::batch<flaot> vc = va + vb;
		vc.store_unaligned(&c[i]);
	}
}
```

Some advanced optimization techniques that can be used alongside explicit SIMD vectorization can be:

**Memory Alignment**: Memory alignment in SIMD is crucial for performance. 
Memory alignment means data addresses must be multiple of certain byte boundaries. 
For example, SSE requires 16-byte alignment, AVX requires 32-byte alignment, 
and AVX-512 requires 64-byte alignment:

```cpp
// potentially misaligned
float* data = new float[100];

// ensure 32-byte alignement for AVX
float* aligned_data = (float*)aligned_alloc(32, 100 * sizeof(float));

// alternative using C++17
std::aligned_storage<sizeof(float[100]), 32> aligned_buffer;
float* aligned_data = reinterpret_cast<float*>(&aligned_buffer);

// misaligned load splits into two operations internally
_mm256_loadu_ps(misaligned_ptr);  // slower

// aligned load uses single operation
_mm256_load_ps(aligned_ptr);  // faster

// potential crashes on processors requiring strict alignment
__m256 data = _mm256_load_ps(misaligned_ptr);

// memory layout considerations
struct alignas(32) aligned_struct {
	__m256 vector_data;  // automatically aligned
	float scalar_data;  // padding added autoamtically
};

// array alignment in class
class vector_processor {
	alignas(32) float data[1024]; // aligned array
};

// cache line alignment, 64-byte is typical cache line
// prevent false sharing in multithreaded code
struct alignas(64) thread_data {
	std::atomic<int> counter;
	char padding[60]; // fills cache line
};
```

**Prefetching**: Prefetching is a technique to load data into the cache before it's needed. 
There are key considerations:
- Prefetch distance: Too short = late, Too long = cache pollution
- Cache line size: Usually 64-byte on x86.
- Bandwidth: Excessive prefetching can saturate memory bandwidth.
- Branch prediction: Only prefetch likely-to-be-used data.

```cpp
// sequential access pattern, CPU will automatically prefetch
for (int i = 0; i < n; i++) {
	// hardware detects pattern
	apply_func(data[i]);
}

// x86 prefetch intrinsics
_mm_prefetch(address, _MM_HINT_T0);   // L1 cache
_mm_prefetch(address, _MM_HINT_T1);   // L2 cache
_mm_prefetch(address, _MM_HINT_T2);   // L2 cache but lower priority
_mm_prefetch(address, _MM_HINT_NTA);  // non-temporal, bypass cache hierarchy

for (int i = 0; i < n; i++) {
	// prefetch 16 elements ahead
	_mm_prefetch(&data[i + 16], _MM_HINT_T0);
	apply_func(data[i]);
}
```

**Data layout**: The layout of data can significantly affect the performance by affecting the memory access patterns:

```cpp
// Array-of-Structure (AoS), poor for vectorization
struct Particle {
	double x, y, z;
	double vx, vy, vz;
};
std::vector<Particle> particles;

// Struct-of-Array (SoA), better for vectorization
struct ParticleSystem {
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
	
	void update_velocities() {
		for (size_t i = 0; i < x.size(); i += 8) {
			__m256 vx_vec = _mm256_load_pd(&vx[i]);
			__m256 vy_vec = _mm256_load_pd(&vy[i]);
			__m256 vz_vec = _mm256_load_pd(&vz[i]);
			// process 8 particles at once
		}
	}
};

// Array-of-Structure-of-Array (AoSoA), balanced approach
struct alignas(32) ParticleChunk {
	double x[8], y[8], z[8];  // SIMD-friendly chunks
	double vx[8], vz[8], vz[8];
};
```
The key advantage of SoA is contiguous memory access for SIMD operations, reducing cache misses, and enabling efficient vectorization.

**Cross-platform vectorization**: If need be we can use abstraction layers for portable SIMD code:

```cpp
// example vector abstraction layer
template<typename T, size_t N>
class SIMDVector {
private:
	#if defined(__AVX__)
		__m256 _data;
	#elif defined(__ARM_NEON)
		float32x4_t _data;
	#else
		std::array<T, N> _data;
	#endif // __AVX__

public:
	// platform independent interface
	void load(const T* ptr) {
		#if defined(__AVX__)
			_data = _mm256_load_ps(ptr);
		#elif defined(__ARM_NEON)
			_data = vld1q_f32(ptr);
		#else
			std::copy(ptr, ptr + N, _data.begin());
		#endif // __AVX__
	}

	void add(const SIMDVector& other) {
		#if defined(__AVX__)
			_data = _mm256_add_ps(_data, other._data);
		#elif defined(__ARM_NEON)
			_data = vaddq_f32(_data, other._data);
		#else
			for (size_t i = 0; i < N; i++) {
				_data[i] += other._data[i];
			}
		#endif // __AVX__
	}
};

// feature detection at runtime
void init_vector_support() {
	#if defined(__x86_64__) || defined(__i386__)
		// use GCC builtin feature detection capabilities
		if (__builtin_cpu_supports("avx2")) {
			// use AVX2 path
		}
		else if (__builtin_cpu_supports("sse4.2")) {
			// fallback to SSE4.2
		}
	#endif // __x86_64__ || __i386__
}
```

**Memory vs. compute bound**: Vector operations are often memory-bandwidth limited. Key metrics include:
- Arithmetic intensity: FLOPs/byte of memory traffic.
- Vector register pressure.
- Cache line utilization.

**Branch prediction**: Using predication instead of branching:
```cpp
// using predication instead of branching
__m256 mask = _mm256_cmp_ps(v1, v2, _CMP_GT_OS);
result = _mm256_blendv_ps(then_value, else_value, mask);
```

**Compiler reports**: Compiler reports are often useful in identifying underlying transformations:

```bash
# GCC
g++ -fopt-info-vec -fopt-info-vec-missed
# Clang
clang++ -Rpass=loop-vectorize -Rpass-missed=loop-vectorize
```

**Performance metrics**: Key metrics to monitor SIMD instructions executed, cache misses, memory bandwidth, 
and vector register utilization. An example of working with performance counters for SIMD optimizations:

```cpp
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

static long perf_event_open(struct perf_event_attr* hw_event, 
							pid_t pid,
							int cpu,
							int group_fd,
							unsigned long flags) {
	return syscall(__NR_pref_event_open, hw_event, pid, cpu, group_fd, flags);
}

// SIMD-specific counter setup
struct perf_event_attr pe;
memset(&pe, 9, sizeof(pe));
pe.type = PERF_TYPE_HARDWARE;
pe.size = sizeof(pe);
pe.config = PERF_COUNT_HW_INSTRUCTIONS;  // count instructions
pe.disabled = 1;
pe.exclude_kernel = 1;

// usage
int fd = perf_event_open(&pe, 0, -1, -1, 0);
ioctl(fd, PERF_EVENT_IOC_RESET, 0);
ioctl(fd, PREF_EVENT_IOC_ENABLE, 0);

// SIMD-enabled code block
vector_operations();

// reading results
long long count;
read(fd, &count, sizeof(count));
```
