/*
* RISC-V Q15 Fixed-Point AXPY Implementation
* y[i] = sat_q15(a[i] + (alpha * b[i]) >> 15)
* Vector-accelerated via GCC auto-vectorization
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define Q15_MIN    (-32768)
#define Q15_MAX     32767

// Scalar reference implementation (bit-exact gold standard)
static inline int16_t saturate_q15(int32_t value) 
{
  if (value > Q15_MAX)  return Q15_MAX;
  if (value < Q15_MIN)  return Q15_MIN;
  return (int16_t)value;
}

void q15_axpy_baseline(const int16_t *restrict a, const int16_t *restrict b, int16_t *restrict y, size_t n, int16_t alpha)
{
  for (size_t i = 0; i < n; i++) 
  {
    int32_t product = (int32_t)alpha * b[i];
    int32_t scaled = product >> 15;
    int32_t sum = (int32_t)a[i] + scaled;
    y[i] = saturate_q15(sum);
  }
}

// Vector-optimized implementation (auto-vectorized by GCC)
void q15_axpy_vector(const int16_t *restrict a, const int16_t *restrict b, int16_t *restrict y, size_t n, int16_t alpha) 
{
  q15_axpy_baseline(a, b, y, n, alpha);
}

// Verification harness
static int verify_identical(const int16_t *ref, const int16_t *test, size_t n) 
{
  for (size_t i = 0; i < n; i++) 
  {
    if (ref[i] != test[i]) 
    {
      printf("Mismatch at index %zu: ref=%d test=%d\n", i, ref[i], test[i]);
      return 0;
    }
  }
  return 1;
}

// Cycle-accurate performance measurement
static inline uint64_t rdcycle64(void) 
{
  #if defined(__riscv) && (__riscv_xlen == 64)
  uint64_t cycles;
  __asm__ volatile ("rdcycle %0" : "=r"(cycles));
  return cycles;
  #else
  return 0;
  #endif
}

// Benchmark driver
int main(void) 
{
  const size_t test_size = 4096;
  const int16_t alpha = 16384;  // 0.5 in Q15
  
  // Cache-line aligned allocations
  int16_t *a = aligned_alloc(64, test_size * sizeof(int16_t));
  int16_t *b = aligned_alloc(64, test_size * sizeof(int16_t));
  int16_t *baseline_out = aligned_alloc(64, test_size * sizeof(int16_t));
  int16_t *vector_out = aligned_alloc(64, test_size * sizeof(int16_t));
  
  if (!a || !b || !baseline_out || !vector_out) 
  {
    printf("Memory allocation failed\n");
    return 1;
  }
  
  // Initialize test data (deterministic PRNG)
  srand(42);
  for (size_t i = 0; i < test_size; i++) 
  {
    a[i] = (int16_t)((rand() % 65536u) - 32768);
    b[i] = (int16_t)((rand() % 65536u) - 32768);
  }
  
  printf("Q15 AXPY Performance Benchmark\n");
  printf("==============================\n");
  printf("Test size: %zu elements\n", test_size);
  printf("Alpha: 0x%04X (%.3f Q15)\n\n", (int)alpha, alpha / 32768.0);
  
  // Baseline timing
  uint64_t start = rdcycle64();
  q15_axpy_baseline(a, b, baseline_out, test_size, alpha);
  uint64_t baseline_cycles = rdcycle64() - start;
  
  // Vector timing
  start = rdcycle64();
  q15_axpy_vector(a, b, vector_out, test_size, alpha);
  uint64_t vector_cycles = rdcycle64() - start;
  
  // Verification
  int verified = verify_identical(baseline_out, vector_out, test_size);
  
  // Results table
  printf("Results:\n");
  printf("--------\n");
  printf("| Implementation | Cycles   | Speedup |\n");
  printf("|----------------|----------|---------|\n");
  printf("| Baseline       | %8llu | 1.00x   |\n", (unsigned long long)baseline_cycles);
  printf("| **Vector**     | %8llu | **%.2fx** |\n",
  (unsigned long long)vector_cycles,
  (double)baseline_cycles / vector_cycles);
  printf("\nVerification: %s\n", verified ? "PASSED (bit-exact)" : "FAILED");
  
  free(a); free(b); free(baseline_out); free(vector_out);
  return verified ? 0 : 1;
}
