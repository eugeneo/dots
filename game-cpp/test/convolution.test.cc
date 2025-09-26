#include "src/convolution.h"

#include <array>
#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/functional/any_invocable.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"           // IWYU pragma: keep
#include "absl/strings/str_join.h"  // IWYU pragma: keep

using namespace uchen::convolution::implementation;

constexpr std::array<float, 12> kPrimes = {2,  3,  5,  7,  11, 13,
                                           17, 19, 23, 29, 31, 37};

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
  std::array<float, 1> output alignas(16);
  Conv2d(input, output, weights, 3,
         ConvolutionOptions{.input_channels = 4,
                            .output_channels = 1,
                            .kernel_height = 3,
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
  std::array<float, 9> output alignas(16);
  Conv2d(input, output, weights, 5,
         ConvolutionOptions{.input_channels = 4,
                            .output_channels = 1,
                            .kernel_height = 3,
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
  std::array<float, 2> output alignas(16);
  Conv2d(input, output, weights, 3,
         ConvolutionOptions{.input_channels = 4,
                            .output_channels = 2,
                            .kernel_height = 3,
                            .kernel_width = 3,
                            .padding_height = 0,
                            .padding_width = 0});
  EXPECT_THAT(output, ::testing::ElementsAre(36, 72));
}

TEST(ConvolutionTest, HorizontalPad) {
  std::array input alignas(16) =
      FillTensor<4, 5, 5>([](size_t, size_t, size_t c) { return c + 1; });
  std::array<float, 8 * 5 * 5> weights alignas(16);
  std::fill(weights.begin(), weights.begin() + weights.size() / 2, 1);
  std::fill(weights.begin() + weights.size() / 2, weights.end(), 2);
  std::array<float, 10> output alignas(16);
  Conv2d(input, output, weights, 5,
         ConvolutionOptions{.input_channels = 4,
                            .output_channels = 2,
                            .kernel_height = 5,
                            .kernel_width = 5,
                            .padding_height = 0,
                            .padding_width = 2});
  EXPECT_THAT(output, ::testing::ElementsAre(120, 240, 200, 400, 300, 600, 280,
                                             560, 240, 480));
}

TEST(ConvolutionTest, VerticalScan) {
  std::array input alignas(16) =
      FillTensor<4, 5, 5>([](size_t, size_t r, size_t) { return r + 1; });
  std::array<float, 8 * 5 * 5> weights alignas(16);
  std::fill(weights.begin(), weights.begin() + weights.size() / 2, 1);
  std::fill(weights.begin() + weights.size() / 2, weights.end(), 2);
  std::array<float, 2> output alignas(16);
  Conv2d(input, output, weights, 5,
         ConvolutionOptions{.input_channels = 4,
                            .output_channels = 2,
                            .kernel_height = 5,
                            .kernel_width = 5,
                            .padding_height = 0,
                            .padding_width = 0});
  EXPECT_THAT(output, ::testing::ElementsAre(300, 600));
}

TEST(ConvolutionTest, VerticalScanWithPad) {
  std::array input alignas(16) =
      FillTensor<4, 5, 5>([](size_t, size_t r, size_t) { return r + 1; });
  std::array<float, 8 * 5 * 5> weights alignas(16);
  std::fill(weights.begin(), weights.begin() + weights.size() / 2, 1);
  std::fill(weights.begin() + weights.size() / 2, weights.end(), 2);
  std::array<float, 10> output alignas(16);
  Conv2d(input, output, weights, 5,
         ConvolutionOptions{.input_channels = 4,
                            .output_channels = 2,
                            .kernel_height = 5,
                            .kernel_width = 5,
                            .padding_height = 2,
                            .padding_width = 0});
  EXPECT_THAT(output, ::testing::ElementsAre(120, 240, 200, 400, 300, 600, 200,
                                             400, 120, 240));
}

TEST(ConvolutionTest, PaddedOnAllSides) {
  std::array input alignas(16) =
      FillTensor<4, 5, 5>([](size_t c, size_t r, size_t co) {
        return c == 0 ? (r) * 5 + co + 1 : 0;
      });
  std::array<float, 8 * 5 * 5> weights alignas(16);
  std::fill(weights.begin(), weights.begin() + weights.size() / 2, 1);
  std::fill(weights.begin() + weights.size() / 2, weights.end(), 2);
  std::array<float, 50> output alignas(16);
  std::array<float, 50> expected;
  expected.fill(0.f);
  std::array<float, 25> expectations = {
      63,  90,  120, 102, 81,  114, 160, 210, 176, 138, 180, 250, 325,
      270, 210, 174, 240, 210, 256, 198, 153, 210, 120, 222, 171};
  for (size_t i = 0; i < expectations.size(); ++i) {
    expected[i * 2] = expectations[i];
    expected[i * 2 + 1] = expectations[i] * 2;
  }

  Conv2d(input, output, weights, 5,
         ConvolutionOptions{.input_channels = 4,
                            .output_channels = 2,
                            .kernel_height = 5,
                            .kernel_width = 5,
                            .padding_height = 2,
                            .padding_width = 2});
  EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
}

TEST(ConvolutionTest, PrimesSquared) {
  std::array input alignas(16) =
      FillTensor<4, 3, 3>([](size_t c, size_t r, size_t co) {
        return c == 0 ? kPrimes[r * 3 + co] : 0;
      });
  std::array<float, 4 * 3 * 3> kernel alignas(16);
  kernel.fill(0);
  for (size_t i = 0; i < kernel.size() / 4; ++i) {
    CHECK_LT(i * 4, kernel.size());
    kernel[i * 4] = kPrimes[i];
  }
  std::array<float, 9> output alignas(16);
  Conv2d(input, output, kernel, 3,
         {.input_channels = 4,
          .output_channels = 1,
          .kernel_height = 3,
          .kernel_width = 3,
          .padding_height = 1,
          .padding_width = 1});
  EXPECT_THAT(
      output,
      ::testing::ElementsAre(
          447, 739, 510, 1001, 1556,
          input[4 * 1] * kernel[4 * 0] + input[4 * 2] * kernel[4 * 1] +
              input[4 * 4] * kernel[4 * 3] + input[4 * 5] * kernel[4 * 4] +
              input[4 * 7] * kernel[4 * 6] + input[4 * 8] * kernel[4 * 7],
          510, 377, 447));
}

TEST(ConvolutionTest, PrimesAreChannels) {
  std::array input alignas(16) =
      FillTensor<4, 3, 3>([](size_t c, size_t r, size_t co) {
        return co == 0 ? kPrimes[r * 4 + c] : 0;
      });
  std::array<float, 4 * 3 * 3> kernel alignas(16);
  kernel.fill(0);
  for (auto r = 0; r < 3; ++r) {
    std::copy(kPrimes.begin() + r * 4, kPrimes.begin() + (r + 1) * 4,
              kernel.begin() + r * 12);
  }
  std::array<float, 9> output alignas(16);
  Conv2d(input, output, kernel, 3,
         {.input_channels = 4,
          .output_channels = 1,
          .kernel_height = 3,
          .kernel_width = 3,
          .padding_height = 1,
          .padding_width = 1});
  EXPECT_THAT(output, ::testing::ElementsAre(
                          0, 2139, 0, 0,
                          std::reduce(kPrimes.begin(), kPrimes.end(), 0,
                                      [](auto a, auto b) { return a + b * b; }),
                          0, 0, 1027, 0));
}

TEST(ConvolutionTest, ThreeXThreeOnFourByFour) {
  std::array input alignas(16) = FillTensor<4, 4, 4>(
      [](size_t ch, size_t, size_t c) { return ch == 0 ? c + 1 : 0; });
  std::array<float, 8 * 3 * 3> weights alignas(16);
  std::fill(weights.begin(), weights.begin() + weights.size() / 2, 1);
  std::fill(weights.begin() + weights.size() / 2, weights.end(), 2);
  std::array<float, 8> output alignas(16);
  Conv2d(input, output, weights, 4,
         ConvolutionOptions{.input_channels = 4,
                            .output_channels = 2,
                            .kernel_height = 3,
                            .kernel_width = 3,
                            .padding_height = 0,
                            .padding_width = 0});
  EXPECT_THAT(output, ::testing::ElementsAre(18, 36, 27, 54, 18, 36, 27, 54));
}

TEST(ConvolutionTest, EightChannels) {
  std::array input alignas(16) =
      FillTensor<8, 3, 3>([](size_t ch, size_t, size_t c) { return ch; });
  std::array<float, 16 * 3 * 3> weights alignas(16);
  std::fill(weights.begin(), weights.begin() + weights.size() / 2, 1);
  std::fill(weights.begin() + weights.size() / 2, weights.end(), 2);
  std::array<float, 2> output alignas(16);
  Conv2d(input, output, weights, 3,
         ConvolutionOptions{.input_channels = 8,
                            .output_channels = 2,
                            .kernel_height = 3,
                            .kernel_width = 3,
                            .padding_height = 0,
                            .padding_width = 0});
  EXPECT_THAT(output, ::testing::ElementsAre(252, 504));
}

TEST(ConvolutionTest, EightChannelsMultipleCellsAndPadding) {
  std::array input alignas(16) =
      FillTensor<8, 4, 4>([](size_t ch, size_t, size_t c) { return ch; });
  std::array<float, 8 * 3 * 3> weights alignas(16);
  weights.fill(1);
  std::array<float, 16> output alignas(16);
  Conv2d(input, output, weights, 4,
         ConvolutionOptions{.input_channels = 8,
                            .output_channels = 1,
                            .kernel_height = 3,
                            .kernel_width = 3,
                            .padding_height = 1,
                            .padding_width = 1});
  EXPECT_THAT(output,
              ::testing::ElementsAre(112, 168, 168, 112, 168, 252, 252, 168,
                                     168, 252, 252, 168, 112, 168, 168, 112));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  return RUN_ALL_TESTS();
}