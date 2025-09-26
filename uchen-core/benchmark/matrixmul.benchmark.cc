#include <sys/types.h>

#include <array>
#include <cmath>
#include <numeric>

#include <benchmark/benchmark.h>

#include "absl/log/check.h"  // IWYU pragma: keep
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"  // IWYU pragma: keep

#include "uchen/math/primitives.h"

namespace uchen::math {
namespace {

template <uint32_t N>
float InlineDotProduct(std::span<const float> a, std::span<const float> b) {
  std::array<float, 32> sum;
  sum.fill(0);
  for (size_t i = 0; i < N; ++i) {
    sum[i % sum.size()] += a[i] * b[i];
  }
  return std::reduce(sum.begin(), sum.end());
}

void BM_InlinedDotProduct(::benchmark::State& state) {
  constexpr size_t N = 1000003;
  std::vector<float> a(N, 1.0f);
  std::vector<float> b(N, 2.0f);
  float sum = 0.0f;
  for (auto _ : state) {
    float sum = InlineDotProduct<N>(a, b);
    ::benchmark::DoNotOptimize(sum);
  }
  ::benchmark::DoNotOptimize(sum);
}

void BM_HighwayDotProduct(::benchmark::State& state) {
  constexpr size_t N = 1000003;
  std::vector<float> a(N, 1.0f);
  std::vector<float> b(N, 2.0f);
  b.front() = 1e6;
  b.back() = 2e6;
  float sum = 0.0f;
  for (auto _ : state) {
    float sum = DotProduct(a, b);
    ::benchmark::DoNotOptimize(sum);
  }
  ::benchmark::DoNotOptimize(sum);
}

BENCHMARK(BM_InlinedDotProduct);
BENCHMARK(BM_HighwayDotProduct);

}  // namespace
}  // namespace uchen::math

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}