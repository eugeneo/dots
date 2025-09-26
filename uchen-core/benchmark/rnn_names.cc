#include "experimental/rnn_names.h"

#include <filesystem>
#include <fstream>

#include <benchmark/benchmark.h>

#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"

#include "datasets/names/names.h"
#include "uchen/layers.h"
#include "uchen/parameters.h"
#include "uchen/training/rnn.h"

std::vector<std::string> names;

template <typename Model>
void CheckParameters(const uchen::ModelParameters<Model>& parameters) {
  size_t nans = 0;
  size_t infs = 0;
  size_t zeroes = 0;
  for (float p : parameters) {
    if (std::isnan(p)) {
      ++nans;
    } else if (std::isinf(p)) {
      ++infs;
    } else if (p == 0) {
      ++zeroes;
    }
  }
  std::cout << "NaNs: " << nans << ", infs: " << infs << ", zeroes: " << zeroes
            << "\n";
}

void CheckGradients(const std::valarray<float>& gradiends) {
  std::cout << gradiends.size() << " values:\n";
  bool first = true;
  for (size_t i = 0; i < std::min(gradiends.size(), static_cast<size_t>(100));
       ++i) {
    if (!first) {
      std::cout << ", ";
    }
    std::cout << gradiends[i];
    first = false;
  }
  std::cout << "\n";
}

// static void BM_RnnNames(benchmark::State& state) {
//   auto dataset = uchen::experimental::PrepareTrainingDataSet(names);
//   size_t split = dataset.size() * .8;
//   uchen::ModelParameters parameters =
//       uchen::RandomParameters(&uchen::experimental::kNameRnn, -1, 1, 5);
//   uchen::training::brainsurgeon::RootInterceptor interceptor("rnn");
//   uchen::training::GradientDescent gd(&uchen::experimental::kNameRnn,
//                                       std::span(dataset).subspan(0, 1),
//                                       dataset.size(), &interceptor);
//   gd.set_parameters(parameters);
//   for (auto _ : state) {
//     gd.Step(std::span(dataset).subspan(split), 0.001, CheckGradients);
//     CheckParameters(gd.parameters());
//     absl::Status status = interceptor.status();
//     if (!status.ok()) {
//       LOG(FATAL) << status.message();
//     }
//     std::cout << "\n";
//   }
// }

// BENCHMARK(BM_RnnNames);

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  auto data_file =
      std::filesystem::current_path() / "benchmark/NationalNames.csv";
  names = uchen::datasets::ReadNamesDb();
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}