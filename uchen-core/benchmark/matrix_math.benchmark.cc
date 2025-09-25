#include <cstddef>
#include <memory>

#include <benchmark/benchmark.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"  // IWYU pragma: keep

#include "uchen/math/matrix.h"

namespace uchen::math {

float Identity(size_t i) { return i; }

namespace {

void BM_SpeedOfLight(benchmark::State& state) {
  constexpr size_t N = 1024 * 1024;
  std::vector a(N, 1.f);
  std::vector c(N, 0.f);
  for (auto _ : state) {
    for (size_t i = 0; i < N; ++i) {
      c[i] = a[i] + Identity(i);
    }
    benchmark::DoNotOptimize(c[20]);
  }
}

void BM_Matrix(benchmark::State& state) {
  constexpr size_t N = 1024;
  RowMajorMatrix<N, N> a = FnMatrix<N, N>([](size_t i) { return 1.f; });
  auto b = FnMatrix<N, N>(&Identity);
  auto c = std::make_unique<RowMajorMatrix<N, N>>();
  for (auto _ : state) {
    *c = a + b;
    benchmark::DoNotOptimize(c->GetRowMajor(20));
  }
}

/*
GCC:
BM_MulRowByColumnToRow          1182459 ns      1250356 ns          560
Clang:
BM_MulRowByColumnToRow          1105331 ns      1169472 ns          602
*/
void BM_MulRowByColumnToRow(benchmark::State& state) {
  constexpr size_t N = 256;
  RowMajorMatrix<N, N> a =
      FnMatrix<N, N>([&](size_t i) { return static_cast<float>(i); });
  ColumnMajorMatrix<N, N> b = FnMatrix<N, N>(
      [&](size_t r, size_t c) { return static_cast<float>(r == c); });
  for (auto _ : state) {
    RowMajorMatrix<N, N> c = a * b;
    benchmark::DoNotOptimize(c.GetRowMajor(20));
  }
}

/*
GCC:
BM_MulRowToColumnToColumn       1162824 ns      1229534 ns          561
Clang:
BM_MulRowToColumnToColumn       1089948 ns      1153197 ns          588
*/
void BM_MulRowToColumnToColumn(benchmark::State& state) {
  constexpr size_t N = 256;
  RowMajorMatrix<N, N> a =
      FnMatrix<N, N>([&](size_t i) { return static_cast<float>(i); });
  ColumnMajorMatrix<N, N> b = FnMatrix<N, N>(
      [&](size_t r, size_t c) { return static_cast<float>(r == c); });
  for (auto _ : state) {
    ColumnMajorMatrix<N, N> c = a * b;
    benchmark::DoNotOptimize(c.GetRowMajor(20));
  }
}

/*
GCC:
BM_MulColumnToRowToColumn       1374925 ns      1453787 ns          507
Clang:
BM_MulColumnToRowToColumn        901668 ns       954437 ns          728
*/
void BM_MulColumnToRowToColumn(benchmark::State& state) {
  constexpr size_t N = 255;
  ColumnMajorMatrix<N, N> a =
      FnMatrix<N, N>([&](size_t i) { return static_cast<float>(i); });
  RowMajorMatrix<N, N> b = FnMatrix<N, N>(
      [&](size_t r, size_t c) { return static_cast<float>(r == c); });
  for (auto _ : state) {
    ColumnMajorMatrix<N, N> c = a * b;
    benchmark::DoNotOptimize(c.GetRowMajor(20));
  }
}

/*
GCC:
BM_MulColumnToColumnToColumn   18561685 ns     18876014 ns           37
Clang:
BM_MulColumnToColumnToColumn   13810943 ns     14042542 ns           50
*/
void BM_MulColumnToColumnToColumn(benchmark::State& state) {
  constexpr size_t N = 255;
  ColumnMajorMatrix<N, N> a =
      FnMatrix<N, N>([&](size_t i) { return static_cast<float>(i); });
  ColumnMajorMatrix<N, N> b = FnMatrix<N, N>(
      [&](size_t r, size_t c) { return static_cast<float>(r == c); });
  for (auto _ : state) {
    ColumnMajorMatrix<N, N> c = a * b;
    benchmark::DoNotOptimize(c.GetRowMajor(20));
  }
}

//
// clang
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// BM_SpeedOfLight                  896411 ns       948423 ns          724
// BM_Matrix                       3212312 ns      3398692 ns          204
//

//
// GCC
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// BM_SpeedOfLight                  797635 ns       843441 ns          819
// BM_Matrix                       3037714 ns      3212001 ns          217
//
BENCHMARK(BM_SpeedOfLight);
BENCHMARK(BM_Matrix);
BENCHMARK(BM_MulRowByColumnToRow);
BENCHMARK(BM_MulRowToColumnToColumn);
BENCHMARK(BM_MulColumnToRowToColumn);
BENCHMARK(BM_MulColumnToColumnToColumn);

}  // namespace
}  // namespace uchen::math

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}