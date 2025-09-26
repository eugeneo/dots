#include "uchen/training/backprop.h"

#include <array>
#include <span>
#include <string>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/test_lib.h"
#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/vector.h"

namespace uchen::training::testing {

namespace {

TEST(Backprop, LinearClear) {
  Model<Linear<Vector<float, 2>, 2>> model;
  ModelParameters parameters(
      &model, ::uchen::testing::RearrangeLinear({{2, 0, 1}, {0, 1, 1}}));
  std::array<float, 6> parameter_gradients;
  EXPECT_THAT(ComputeGradients(model.layer<0>(), {2, 3}, {5, 6},
                               parameters.layer_parameters<0>(),
                               parameter_gradients, nullptr),
              ::testing::ElementsAre(10, 6));
  EXPECT_THAT(
      parameter_gradients,
      ::testing::ElementsAre(::testing::FloatEq(5), ::testing::FloatEq(6),
                             ::testing::FloatEq(10), ::testing::FloatEq(12),
                             ::testing::FloatEq(15), ::testing::FloatEq(18)));
}

TEST(Backprop, LinearParameter) {
  Model<Linear<Vector<float, 2>, 3>> model;
  ModelParameters parameters(&model, 0);
  std::array<float, model.all_parameters_count()> parameter_gradients;
  EXPECT_THAT(ComputeGradients(model.layer<0>(), {3, -2}, {2.f, 2.5f, 3.f},
                               parameters.layer_parameters<0>(),
                               parameter_gradients, nullptr),
              ::testing::ElementsAre(0, 0));
  EXPECT_THAT(parameter_gradients, ::testing::ElementsAre(2, 2.5f, 3.f, 6, 7.5f,
                                                          9.f, -4, -5.f, -6.f));
}

TEST(Backprop, LinearSimpleParameters) {
  Model<Linear<Vector<float, 1>, 1>> model;
  ModelParameters parameters(&model, 0);
  std::array<float, model.all_parameters_count()> parameter_gradients;
  EXPECT_THAT(ComputeGradients(model.layer<0>(), {5}, {5},
                               parameters.layer_parameters<0>(),
                               parameter_gradients, nullptr),
              ::testing::ElementsAre(0));
  EXPECT_THAT(parameter_gradients, ::testing::ElementsAre(5, 25));
}

TEST(Backprop, LinearSimpleInputs) {
  Model<Linear<Vector<float, 1>, 1>> model;
  ModelParameters parameters(&model, 5);
  std::array<float, model.all_parameters_count()> parameter_gradients;
  EXPECT_THAT(ComputeGradients(model.layer<0>(), {3}, {5},
                               parameters.layer_parameters<0>(),
                               parameter_gradients, nullptr),
              ::testing::ElementsAre(25));
  EXPECT_THAT(parameter_gradients, ::testing::ElementsAre(5, 15));
}

TEST(Backprop, LinearInputs) {
  Model<Linear<Vector<float, 2>, 3>> model;
  ModelParameters parameters = {&model, {-1, 2, -3, -4, 5, -6, 7, -8, 9}};
  std::array<float, model.all_parameters_count()> parameter_gradients;
  EXPECT_THAT(ComputeGradients(model.layer<0>(), {}, {0.1, -0.2, 0.3},
                               parameters.layer_parameters<0>(),
                               parameter_gradients, nullptr),
              ::testing::ElementsAre(-3.2, 5));
  EXPECT_THAT(parameter_gradients,
              ::testing::ElementsAre(0.1, -0.2, 0.3, 0, 0, 0, 0, 0, 0));
}

TEST(Backprop, Relu) {
  EXPECT_THAT(
      ComputeGradients(ElementWise<Vector<float, 10>, Relu>(),
                       {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10},
                       {.1, .2, .5, .4, .5, 6, 0, 8, 0, 10}, {}, {}, nullptr),
      ::testing::ElementsAre(0, .2, 0, .4, 0, 6, 0, 8, 0, 10));
}

TEST(Backprop, Softmax1Cat) {
  Categories<std::string, float, 1> layer({"a"});
  auto gradients =
      ComputeGradients(layer, {1}, {1}, {}, std::span<float, 0>(), nullptr);
  EXPECT_THAT(gradients, ::testing::ElementsAre(1));
}

TEST(Backprop, Softmax2Cat) {
  Categories<std::string, float, 3> layer({"a", "b", "c"});
  auto gradients = ComputeGradients(layer, {1, 2, 3}, {1, 2, 3}, {},
                                    std::span<float, 0>(), nullptr);
  // Used PyTorch to confirm the expected values. Don't question them!
  EXPECT_THAT(gradients, ::testing::ElementsAre(1, 2, 3));
}

TEST(Backprop, Sigmoid) {
  Vector<double, 4> v = {0.5, 1, 0, 0.731};
  Model m = layers::Input<decltype(v)> | layers::Sigmoid;
  auto gradients =
      ComputeGradients(m.template layer<1>(), v, {1, 0.2e-10, 3, 4}, {},
                       std::span<float, 0>(), nullptr);
  EXPECT_THAT(gradients, ::testing::ElementsAre(0.25, 0, 0, 0.78655601));
}

}  // namespace
}  // namespace uchen::training::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}