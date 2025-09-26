#include "uchen/text/attention.h"

#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include "uchen/vector.h"

namespace {

/*

Clang:
-----------------------------------------------------------------
Benchmark                       Time             CPU   Iterations
-----------------------------------------------------------------
BM_Attention<1024, 64>    1531933 ns      1671202 ns          413

GCC:
-----------------------------------------------------------------
Benchmark                       Time             CPU   Iterations
-----------------------------------------------------------------
BM_Attention<1024, 64>    1730520 ns      1839777 ns          384

*/
template <size_t T, size_t E>
void BM_Attention(benchmark::State& state) {
  constexpr size_t PC = T * (E + 1) * 3;
  std::vector<float> v_input(T * E);
  std::vector<float> parameters(PC);
  using Attention = uchen::text::impl::AttentionLayerContext<T, E>;
  auto attention = std::make_unique<Attention>();
  uchen::Vector<float, T * E> input(
      std::span<const float>(v_input).template first<T * E>());
  typename Attention::parameters p{
      std::span<const float>(parameters)
          .template first<Attention::parameters::Size>()};
  for (auto _ : state) {
    auto result = attention->Calculate(input, p);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(BM_Attention<1024, 64>);

}  // namespace

BENCHMARK_MAIN();
