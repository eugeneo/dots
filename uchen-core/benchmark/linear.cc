/*
BM_Linear<100, 200>          678 ns          693 ns       983938
BM_Linear<2500, 8>          2513 ns         2567 ns       272289
BM_Linear<8, 2500>          1061 ns         1084 ns       711797
BM_Linear<4000, 2000>    1194393 ns      1218994 ns          495
BM_Linear<1000000, 8>    1666250 ns      1699779 ns          408
BM_Linear<8, 1000000>    1480243 ns      1510032 ns          462

M2
BM_Linear<100, 200>         3192 ns         3186 ns       220937
BM_Linear<2500, 8>          3195 ns         3151 ns       224583
BM_Linear<8, 2500>          1024 ns         1002 ns       698931
BM_Linear<4000, 2000>    5758479 ns      5590016 ns          125
BM_Linear<1000000, 8>    4347109 ns      4330687 ns          163
BM_Linear<8, 1000000>     680755 ns       673311 ns         1064
*/

#include "uchen/linear.h"

#include <array>
#include <cstddef>
#include <iterator>
#include <vector>

#include <benchmark/benchmark.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "uchen/layer_traits.h"
#include "uchen/layers.h"
#include "uchen/model.h"
#include "uchen/parameters.h"

constexpr uchen::Model kModel = uchen::layers::Input<uchen::Vector<float, 1>> |
                                uchen::layers::Linear<1000000000>;
using M = std::remove_cv_t<decltype(kModel)>;
static uchen::ModelParameters<M>* params;

template <size_t Is, size_t Os, typename D = float>
static void BM_Linear(benchmark::State& state) {
  auto ar = std::make_unique<std::array<float, Os>>();
  uchen::internal::InferenceLayerContext ctx(ar.get());
  uchen::Linear<uchen::Vector<D, Is>, Os> layer;
  uchen::Vector<D, Is> input;
  constexpr size_t C =
      uchen::LayerTraits<decltype(layer), decltype(input)>::parameter_count;
  std::vector<float> parameter_data;
  std::copy(params->begin(), params->begin() + C,
            std::back_inserter(parameter_data));
  uchen::Parameters<C> parameters(parameter_data);
  for (auto _ : state) {
    benchmark::DoNotOptimize(layer(input, parameters, &ctx));
  }
}

// BENCHMARK(BM_Linear<98, 200>);
// BENCHMARK(BM_Linear<100, 192>);
// BENCHMARK(BM_Linear<100, 193>);
// BENCHMARK(BM_Linear<100, 196>);
// BENCHMARK(BM_Linear<100, 199>);
// BENCHMARK(BM_Linear<4096, 8>);
// BENCHMARK(BM_Linear<2048, 16>);
// BENCHMARK(BM_Linear<1024, 32>);
// BENCHMARK(BM_Linear<512, 64>);
// BENCHMARK(BM_Linear<256, 128>);
// BENCHMARK(BM_Linear<128, 256>);
// BENCHMARK(BM_Linear<32, 1024>);
// BENCHMARK(BM_Linear<16, 2048>);
BENCHMARK(BM_Linear<100, 200>);
BENCHMARK(BM_Linear<2500, 8>);
BENCHMARK(BM_Linear<8, 2500>);
BENCHMARK(BM_Linear<4000, 2000>);
BENCHMARK(BM_Linear<1000000, 8>);
BENCHMARK(BM_Linear<8, 1000000>);
// BENCHMARK(BM_Linear<20000, 1>);
// BENCHMARK(BM_Linear<1, 10000>);

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  // Unique ptr will cleanup afterwards.
  auto p = uchen::RandomParameters(&kModel, -1, 1, 42);
  params = &p;
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}