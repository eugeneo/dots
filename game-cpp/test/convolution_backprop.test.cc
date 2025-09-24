#include <cstddef>

#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "src/convolution.h"

using uchen::convolution::implementation::Conv2DParameterGradients;

TEST(ParameterGradientTest, OneKernel4OutputNoPadding) {
  std::array<float, 4 * 3 * 3> gradients alignas(16);
  std::array<float, 4> gradient_out alignas(16) = {2, 3, 4, 5};
  std::array<float, 4 * 4 * 4> input alignas(16);
  input.fill(0);
  for (size_t i = 0; i < 16; ++i) {
    input[i * 4] = i + 1;
  }
  Conv2DParameterGradients(gradient_out, input, gradients, 4,
                           {.input_channels = 4,
                            .output_channels = 1,
                            .kernel_height = 3,
                            .kernel_width = 3,
                            .padding_height = 0,
                            .padding_width = 0});
  std::array<float, 4 * 4 * 4> expects;
  expects.fill(0);
  for (size_t i = 0; i < 16; ++i) {
    expects[i * 4] = (i + 1) * 5;
  }
  EXPECT_FLOAT_EQ(gradients[0], 58);
  EXPECT_FLOAT_EQ(gradients[gradients.size() - 4], 198);
}

TEST(ParameterGradientTest, OneKernelPadding) {
  std::array<float, 4 * 3 * 3> gradients alignas(16);
  std::array<float, 9> gradient_out alignas(16);
  std::iota(gradient_out.begin(), gradient_out.end(), 1);
  std::array<float, 4 * 3 * 3> input alignas(16);
  input.fill(0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; j++) {
      input[(i * 3 + j) * 4] = j + 1;
    }
  }
  Conv2DParameterGradients(gradient_out, input, gradients, 3,
                           {.input_channels = 4,
                            .output_channels = 1,
                            .kernel_height = 3,
                            .kernel_width = 3,
                            .padding_height = 1,
                            .padding_width = 1});
  for (size_t i = 1; i < 9; ++i) {
    gradients[i] = gradients[i * 4];
  }
  EXPECT_THAT(std::span(gradients).first(9),
              ::testing::ElementsAre(43, 82, 61, 51, 96, 69, 25, 46, 31));
}

TEST(ParameterGradientTest, OneKernelOneElementPadding) {
  std::array<float, 4 * 3 * 3> gradients alignas(16);
  std::array<float, 1> gradient_out alignas(16) = {1};
  std::array<float, 4> input alignas(16) = {2, 0, 0, 0};
  Conv2DParameterGradients(gradient_out, input, gradients, 1,
                           {.input_channels = 4,
                            .output_channels = 1,
                            .kernel_height = 3,
                            .kernel_width = 3,
                            .padding_height = 1,
                            .padding_width = 1});
  EXPECT_FLOAT_EQ(gradients[16], 2);
  gradients[16] = 0;
  EXPECT_THAT(gradients, ::testing::Each(0));
}

TEST(ParameterGradientTest, TwoKernelOneElementPadding) {
  std::array<float, 8 * 3 * 3> gradients alignas(16);
  std::array<float, 2> gradient_out alignas(16) = {1, 3};
  std::array<float, 4> input alignas(16) = {2, 0, 0, 0};
  Conv2DParameterGradients(gradient_out, input, gradients, 1,
                           {.input_channels = 4,
                            .output_channels = 2,
                            .kernel_height = 3,
                            .kernel_width = 3,
                            .padding_height = 1,
                            .padding_width = 1});
  EXPECT_FLOAT_EQ(gradients[16], 2);
  EXPECT_FLOAT_EQ(gradients[36 + 16], 6);
  gradients[16] = 0;
  gradients[36 + 16] = 0;
  EXPECT_THAT(gradients, ::testing::Each(0));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  return RUN_ALL_TESTS();
}