#include "src/convolution.h"

#include <array>
#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/functional/any_invocable.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"  // IWYU pragma: keep
#include "absl/strings/str_join.h"  // IWYU pragma: keep

#include "gmock/gmock.h"

using namespace uchen::convolution::implementation;

template <size_t c, size_t h, size_t w>
std::array<float, c * h * w> FillTensor(
    const absl::AnyInvocable<float(size_t ch, size_t r, size_t col) const>& f) {
  std::array<float, c * h * w> result;
  for (size_t channel = 0; channel < c; ++channel) {
    for (size_t row = 0; row < h; ++row) {
      for (size_t column = 0; column < w; ++column) {
        result[channel + (column + row * w) * c] = f(channel, row, column);
      }
    }
  }
  return result;
}

TEST(ConvolutionTest, OneElement) {
  std::array<float, 4 * 3 * 3> input alignas(16);
  std::array<float, 4 * 3 * 3> weights alignas(16);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = i + 1;
    weights[i] = 1.f / (i + 1);
  }
  std::array<float, 1> output;
  Conv2dStride4(input, output, weights, 3, 1,
                ConvolutionOptions{.kernel_height = 3,
                                   .kernel_width = 3,
                                   .padding_height = 0,
                                   .padding_width = 0});
  EXPECT_THAT(output, ::testing::Each(::testing::FloatEq(36)));
}

TEST(ConvolutionTest, ThreeByThree) {
  std::array input alignas(16) =
      FillTensor<4, 5, 5>([](size_t, size_t, size_t c) { return c + 1; });
  std::array<float, 4 * 3 * 3> weights alignas(16);
  weights.fill(1.f);
  std::array<float, 9> output;
  Conv2dStride4(input, output, weights, 5, 1,
                ConvolutionOptions{.kernel_height = 3,
                                   .kernel_width = 3,
                                   .padding_height = 0,
                                   .padding_width = 0});
  EXPECT_THAT(output,
              ::testing::ElementsAre(72, 108, 144, 72, 108, 144, 72, 108, 144));
}

TEST(ConvolutionTest, TwoOutputChannels) {
  std::array input alignas(16) =
      FillTensor<4, 3, 3>([](size_t, size_t, size_t c) { return 1; });
  std::array<float, 4 * 3 * 3 * 2> weights alignas(16);
  std::fill(weights.begin(), weights.begin() + weights.size() / 2, 1);
  std::fill(weights.begin() + weights.size() / 2, weights.end(), 2);
  std::array<float, 2> output;
  Conv2dStride4(input, output, weights, 3, 2,
                ConvolutionOptions{.kernel_height = 3,
                                   .kernel_width = 3,
                                   .padding_height = 0,
                                   .padding_width = 0});
  EXPECT_THAT(output, ::testing::ElementsAre(36, 72));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  return RUN_ALL_TESTS();
}