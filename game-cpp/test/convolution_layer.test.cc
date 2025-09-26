#include <array>
#include <cstddef>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"           // IWYU pragma: keep
#include "absl/strings/str_join.h"  // IWYU pragma: keep

#include "gmock/gmock.h"
#include "src/convolution.h"
#include "uchen/layers.h"
#include "uchen/model.h"

using namespace uchen::convolution;

TEST(ConvolutionLayerTest, Forward) {
  ConvolutionInput<4, 5, 5> input;
  std::fill(input.data().begin(), input.data().end(), 0);
  input(1, 2, 3) = 1;
  input(1, 3, 3) = 2;
  input(1, 4, 3) = 3;
  constexpr uchen::Model model =
      uchen::layers::Input<ConvolutionInput<4, 5, 5>> | Conv2d<4, 5, 5, 2, 2>;
  auto parameter_store = uchen::NewFlatStore(&model);
  std::span data = parameter_store->data();
  std::fill(data.begin(), data.end(), 0);
  data[4 * (3 * 5 + 0) + 1 + 100] = 5;
  data[4 * (3 * 5 + 1) + 1 + 100] = -7;
  data[4 * (3 * 5 + 2) + 1 + 100] = 11;
  uchen::ModelParameters parameters{&model, parameter_store};
  EXPECT_EQ(parameters.size(), 400);
  EXPECT_EQ(parameters.layer_parameters<1>().size(), 400);
  auto result = model(input, parameters);
  std::array row2 = {result(1, 0, 2), result(1, 1, 2), result(1, 2, 2),
                     result(1, 3, 2), result(1, 4, 2)};
  EXPECT_THAT(row2, ::testing::ElementsAre(0, 0, 11, 15, 24));
}

TEST(ConvolutionLayerTest, ForwardRelu) {
  ConvolutionInput<4, 3, 3> input;
  std::fill(input.data().begin(), input.data().end(), 0);
  input(0, 0, 1) = 1;
  input(0, 1, 1) = 2;
  input(0, 2, 1) = 3;
  constexpr uchen::Model model =
      uchen::layers::Input<decltype(input)> |
      Conv2dWithFilter<4, 3, 3, 1, 1>(Flatten(ReluFilter()));
  auto parameter_store = uchen::NewFlatStore(&model);
  std::span data = parameter_store->data();
  std::fill(data.begin(), data.end(), 0);
  data[4 * (3 + 0)] = 5;
  data[4 * (3 + 1)] = -7;
  data[4 * (3 + 2)] = 11;
  uchen::ModelParameters parameters{&model, parameter_store};
  auto result = model(input, parameters);
  std::array row2 = {
      result[(0 + 3) * 4],
      result[(1 + 3) * 4],
      result[(2 + 3) * 4],
  };
  EXPECT_THAT(row2, ::testing::ElementsAre(22 - 7, 33 - 14 + 5, 0));
}

TEST(ConvolutionLayerTest, Flatten) {
  Flatten<ReluFilter> filter;
  ConvolutionInput<4, 3, 3> inp;
  auto vector = filter(inp);
  EXPECT_EQ(vector.size(), 36);
  static_assert(
      std::is_same_v<decltype(filter(inp)), uchen::Vector<float, 4 * 3 * 3>>);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  return RUN_ALL_TESTS();
}