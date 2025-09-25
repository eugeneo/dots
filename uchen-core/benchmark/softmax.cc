#include "uchen/math/softmax.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>

#include "uchen/math/matrix.h"

namespace {

constexpr size_t N = 1024;

void BM_Column(benchmark::State& state) {
  std::vector<float> input(N * N, 20);
  std::vector<float> output(N * N, 0);
  auto a = uchen::math::AsColumnMajorView<N, N>(std::span<const float>(input));
  for (auto _ : state) {
    uchen::math::AsColumnMajorView<N, N>(std::span<float>(output)) = Softmax(a);
    benchmark::DoNotOptimize(output[20]);
  }
}

void BM_Row(benchmark::State& state) {
  std::vector<float> input(N * N, 20);
  std::vector<float> output(N * N, 0);
  auto a = uchen::math::AsRowMajorView<N, N>(std::span<const float>(input));
  for (auto _ : state) {
    uchen::math::AsRowMajorView<N, N>(std::span<float>(output)) = Softmax(a);
    benchmark::DoNotOptimize(output[20]);
  }
}

void BM_Cross(benchmark::State& state) {
  std::vector<float> input(N * N, 20);
  std::vector<float> output(N * N, 0);
  auto a = uchen::math::AsRowMajorView<N, N>(std::span<const float>(input));
  for (auto _ : state) {
    uchen::math::AsColumnMajorView<N, N>(std::span<float>(output)) = Softmax(a);
    benchmark::DoNotOptimize(output[20]);
  }
}

/*
GCC:
-----------------------------------------------------
Benchmark           Time             CPU   Iterations
-----------------------------------------------------
BM_Column      947533 ns      1033673 ns          658
BM_Row        3089548 ns      3370420 ns          210
BM_Cross      8790879 ns      9589935 ns           68

Clang:
-----------------------------------------------------
Benchmark           Time             CPU   Iterations
-----------------------------------------------------
BM_Column     1610274 ns      1756506 ns          399
BM_Row        3223952 ns      3514578 ns          197
BM_Cross     10247907 ns     11176722 ns           60

VC++:
-----------------------------------------------------
Benchmark           Time             CPU   Iterations
-----------------------------------------------------
BM_Column     1053070 ns      1045850 ns          747
BM_Row        3218425 ns      3208705 ns          224
BM_Cross     12231350 ns     12187500 ns           50

Clang M2
-----------------------------------------------------
Benchmark           Time             CPU   Iterations
-----------------------------------------------------
BM_Column      979890 ns       979660 ns          686
BM_Row        2762588 ns      2762107 ns          252
BM_Cross      5792393 ns      5791119 ns          126
*/

BENCHMARK(BM_Column);
BENCHMARK(BM_Row);
BENCHMARK(BM_Cross);

}  // namespace

BENCHMARK_MAIN();
