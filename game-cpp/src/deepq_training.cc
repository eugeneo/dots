#include <cstddef>

#include "absl/flags/parse.h"

#include "src/convolution.h"
#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/training/training.h"

using uchen::convolution::Conv2dWithFilter;
using uchen::convolution::ConvolutionInput;
using uchen::convolution::Flatten;
using uchen::convolution::ReluFilter;
using uchen::layers::Linear;
using uchen::layers::Relu;

constexpr uchen::Model ConvQNetwork =
    uchen::layers::Input<ConvolutionInput<4, 64, 64>> |
    Conv2dWithFilter<32, 3, 3, 1, 1>(ReluFilter()) |
    Conv2dWithFilter<64, 3, 3, 1, 1>(ReluFilter()) |
    Conv2dWithFilter<64, 3, 3, 1, 1>(Flatten<ReluFilter>()) | Linear<512> |
    Relu | Linear<64 * 64>;

using QModel = decltype(ConvQNetwork);

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  ConvolutionInput<4, 64, 64> input;
  uchen::ModelParameters params(&ConvQNetwork);
  auto r = ConvQNetwork(input, params);
  uchen::training::Training training(&ConvQNetwork, params);
  uchen::training::TrainingData<QModel::input_t, QModel::output_t> data = {};
  for (size_t generation = 1; training.Loss(data) > 0.0001; ++generation) {
    training = training.Generation(data, 0.001);
    if (generation > 500) {
      LOG(ERROR) << "Taking too long!";
    }
  }

  LOG(INFO) << typeid(r).name() << " " << sizeof(r) << " " << sizeof(params);
  return 0;
}