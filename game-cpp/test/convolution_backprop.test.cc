#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/strings/str_join.h"  // IWYU pragma: keep

#include "gmock/gmock.h"
#include "src/convolution.h"

using uchen::convolution::implementation::Conv2dInputGradients;
using uchen::convolution::implementation::Conv2dParameterGradients;

TEST(Conv2dParameterGradients, OneKernel4OutputNoPadding) {
  std::array<float, 4 * 3 * 3> gradients alignas(16);
  std::array<float, 4> gradient_out alignas(16) = {2, 3, 4, 5};
  std::array<float, 4 * 4 * 4> input alignas(16);
  input.fill(0);
  for (size_t i = 0; i < 16; ++i) {
    input[i * 4] = i + 1;
  }
  Conv2dParameterGradients(gradient_out, input, gradients, 4,
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

TEST(Conv2dParameterGradients, OneKernelPadding) {
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
  Conv2dParameterGradients(gradient_out, input, gradients, 3,
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

TEST(Conv2dParameterGradients, OneKernelOneElementPadding) {
  std::array<float, 4 * 3 * 3> gradients alignas(16);
  std::array<float, 1> gradient_out alignas(16) = {1};
  std::array<float, 4> input alignas(16) = {2, 0, 0, 0};
  Conv2dParameterGradients(gradient_out, input, gradients, 1,
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

TEST(Conv2dParameterGradients, TwoKernelOneElementPadding) {
  std::array<float, 8 * 3 * 3> gradients alignas(16);
  std::array<float, 2> gradient_out alignas(16) = {1, 3};
  std::array<float, 4> input alignas(16) = {2, 0, 0, 0};
  Conv2dParameterGradients(gradient_out, input, gradients, 1,
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

TEST(Conv2dInputGradients, ThreeByThreeKernelFieldNoPad) {
  std::array<float, 4 * 3 * 3> kernel alignas(16);
  for (float ch = 0; ch < 4; ++ch) {
    for (float el = 0; el < 9; ++el) {
      kernel[el * 4 + ch] = ch + 1;
    }
  }
  std::array<float, 1> output_gradient alignas(16) = {3};
  std::array<float, 4 * 3 * 3> receiver alignas(16);
  Conv2dInputGradients(output_gradient, kernel, receiver, 3,
                       {
                           .input_channels = 4,
                           .output_channels = 1,
                           .kernel_height = 3,
                           .kernel_width = 3,
                           .padding_height = 0,
                           .padding_width = 0,
                       });
  for (float el = 0; el < 9; ++el) {
    std::span data = std::span(receiver).subspan(el * 4, 4);
    EXPECT_THAT(data, ::testing::ElementsAre(3, 6, 9, 12)) << "Element " << el;
  }
}

TEST(Conv2dInputGradients, ThreeByThreeKernelFieldNoPadRow) {
  // Each channel fills column
  std::array<float, 4 * 3 * 3> kernel alignas(16);
  kernel.fill(0);
  for (int i = 0; i < 3; ++i) {
    kernel[(i * 3) * 4 + i] = i + 1;
    kernel[(i * 3 + 1) * 4 + i] = i + 1;
    kernel[(i * 3 + 2) * 4 + i] = i + 1;
  }
  std::array<float, 1> output_gradient alignas(16) = {3};
  std::array<float, 4 * 3 * 3> receiver alignas(16);
  Conv2dInputGradients(output_gradient, kernel, receiver, 3,
                       {
                           .input_channels = 4,
                           .output_channels = 1,
                           .kernel_height = 3,
                           .kernel_width = 3,
                           .padding_height = 0,
                           .padding_width = 0,
                       });
  EXPECT_THAT(receiver, ::testing::ElementsAre(

                            3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0,

                            0, 6, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0,

                            0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 9, 0));
}

TEST(Conv2dInputGradients, ThreeByThreeKernelFieldNoPadColumn) {
  // Transpose of the above
  std::array<float, 4 * 3 * 3> kernel alignas(16);
  kernel.fill(0);
  for (int i = 0; i < 3; ++i) {
    kernel[i * 4 + i] = i + 1;
    kernel[(3 + i) * 4 + i] = i + 1;
    kernel[(6 + i) * 4 + i] = i + 1;
  }
  std::array<float, 1> output_gradient alignas(16) = {3};
  std::array<float, 4 * 3 * 3> receiver alignas(16);
  Conv2dInputGradients(output_gradient, kernel, receiver, 3,
                       {
                           .input_channels = 4,
                           .output_channels = 1,
                           .kernel_height = 3,
                           .kernel_width = 3,
                           .padding_height = 0,
                           .padding_width = 0,
                       });
  EXPECT_THAT(receiver, ::testing::ElementsAre(

                            3, 0, 0, 0, 0, 6, 0, 0, 0, 0, 9, 0,

                            3, 0, 0, 0, 0, 6, 0, 0, 0, 0, 9, 0,

                            3, 0, 0, 0, 0, 6, 0, 0, 0, 0, 9, 0));
}

TEST(Conv2dInputGradients, ThreeByThreeKernelTwoKernels) {
  std::array<float, 8 * 3 * 3> kernel alignas(16);
  kernel.fill(0);
  for (int el = 0; el < 9; ++el) {
    kernel[el * 4] = 1;
    kernel[36 + el * 4] = 2;
    kernel[36 + el * 4 + 1] = 3;
  }
  std::array<float, 2> output_gradient alignas(16) = {3, 4};
  std::array<float, 4 * 3 * 3> receiver alignas(16);
  Conv2dInputGradients(output_gradient, kernel, receiver, 3,
                       {
                           .input_channels = 4,
                           .output_channels = 2,
                           .kernel_height = 3,
                           .kernel_width = 3,
                           .padding_height = 0,
                           .padding_width = 0,
                       });
  for (int el = 0; el < 9; ++el) {
    EXPECT_THAT(std::span(receiver).subspan(el * 4, 4),
                ::testing::ElementsAre(11, 12, 0, 0))
        << "Element " << el;
  }
}

TEST(Conv2dInputGradients, FiveByFiveKernelPad) {
  std::array<float, 4 * 5 * 5> kernel alignas(16);
  kernel.fill(0);
  for (int i = 0; i < 25; i++) {
    kernel[i * 4] = 1;
  }
  constexpr int kSize = 7;
  std::array<float, kSize * kSize> output_gradient alignas(16);
  output_gradient.fill(1);
  std::array<float, 4 * kSize * kSize> receiver alignas(16);
  Conv2dInputGradients(output_gradient, kernel, receiver, kSize,
                       {
                           .input_channels = 4,
                           .output_channels = 1,
                           .kernel_height = 5,
                           .kernel_width = 5,
                           .padding_height = 2,
                           .padding_width = 2,
                       });
  for (int i = 1; i < kSize * kSize; ++i) {
    std::swap(receiver[i], receiver[i * 4]);
  }
  std::array<float, 16> quarter = {9,  12, 15, 15, 12, 16, 20, 20,
                                   15, 20, 25, 25, 15, 20, 25, 25};
  std::array<float, output_gradient.size()> expects;
  constexpr int kQuarter = 4;
  for (int r = 0; r < kQuarter; ++r) {
    for (int c = 0; c < kQuarter; ++c) {
      expects[r * kSize + c] = quarter[r * kQuarter + c];
      expects[(6 - r) * kSize + c] = quarter[r * kQuarter + c];
      expects[r * kSize + 6 - c] = quarter[r * kQuarter + c];
      expects[(6 - r) * kSize + (6 - c)] = quarter[r * kQuarter + c];
    }
  }
  EXPECT_THAT(std::span(receiver).first(kSize * kSize),
              ::testing::ElementsAreArray(expects));
  EXPECT_THAT(std::span(receiver).subspan(kSize * kSize), ::testing::Each(0));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  return RUN_ALL_TESTS();
}