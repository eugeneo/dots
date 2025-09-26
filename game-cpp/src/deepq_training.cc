#include <cstddef>

#include "absl/flags/parse.h"

#include "src/convolution.h"
#include "uchen/layers.h"
#include "uchen/linear.h"

/*
Input: [C=3, H, W]  (game grid channels)

Conv2d(32, kernel=3, stride=1, padding=1) → ReLU
Conv2d(64, kernel=3, stride=1, padding=1) → ReLU
Conv2d(64, kernel=3, stride=1, padding=1) → ReLU
Flatten →
FC(512) → ReLU
FC(height*width)  # Q-values for each board position
*/

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

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  ConvolutionInput<4, 64, 64> input;
  uchen::ModelParameters params(&ConvQNetwork);
  auto r = ConvQNetwork(input, params);
  LOG(INFO) << typeid(r).name() << " " << sizeof(r) << " " << sizeof(params);
  return 0;
}