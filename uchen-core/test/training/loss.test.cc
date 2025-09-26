#include "uchen/training/loss.h"

#include <array>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/test_lib.h"
#include "uchen/layers.h"

namespace uchen::training::testing {

namespace {

TEST(SquaredLoss, Forward) {
  SquaredLoss<float, 3> l;
  auto result = l.Loss({-1.5, 0, 2.5}, {0.5, 1, 2});
  EXPECT_FLOAT_EQ(result, (4 + 1 + 0.25) / 3);
}

TEST(SquaredLoss, Backward) {
  auto result =
      SquaredLoss<float, 4>().Gradient({-1.5, 0, 2.5, 0}, {0.5, 1, 2, 0});
  EXPECT_THAT(result, ::testing::ElementsAre(-4, -2, 1, 0));
}

TEST(CrossEntropy, Case1) {
  CrossEntropy<std::string_view> loss;
  CategoricalResult<std::string_view, 2> result({"a", "b"}, {0.5f, 0.5f});
  EXPECT_FLOAT_EQ(loss.Loss(result, "a"), 0.693147);
  auto gradients = loss.Gradient(result, "a");
  EXPECT_THAT(std::vector(std::begin(gradients), std::end(gradients)),
              ::testing::ElementsAre(-0.5, 0.5));
}

TEST(CrossEntropy, Case2) {
  Categories<std::string, float, 4> cats({"a", "b", "c", "d"});
  CrossEntropy<std::string_view> loss;
  CategoricalResult<std::string_view, 4> arg({"a", "b", "c", "d"},
                                             {0, 0, 1, 0});
  EXPECT_FLOAT_EQ(loss.Loss(arg, "b"), 1.7436684);
  auto loss_gradient = loss.Gradient(arg, "b");
  EXPECT_THAT(std::vector(std::begin(loss_gradient), std::end(loss_gradient)),
              ::testing::ElementsAre(::testing::FloatEq(0.174877703),
                                     ::testing::FloatEq(-0.825122297),
                                     ::testing::FloatEq(0.47536689),
                                     ::testing::FloatEq(0.174877703)));
}

// This is inspired by https://www.youtube.com/watch?v=xBEh66V9gZo&t=364s
TEST(GradTest, EntropyLossFromVideo) {
  using std::string_view_literals::operator""sv;
  Model model =
      layers::Input<Vector<float, 2>> | layers::Linear<2> | layers::Relu |
      layers::Categories(std::array{"Setosa"sv, "Versicolor"sv, "Virginica"sv});
  std::vector params1 =
      ::uchen::testing::RearrangeLinear({{-2.5, 0.6, 1.6}, {-1.5, 0.4, 0.7}});
  std::vector params2 = ::uchen::testing::RearrangeLinear(
      {{-0.1, 1.5, -2}, {2.4, -5.2, 0}, {-2.2, 3.7, 1}});
  params1.insert(params1.end(), params2.begin(), params2.end());
  ModelParameters parameters(&model, params1);
  CrossEntropy<std::string_view> loss_fn;

  CategoricalResult result = model({0.04, 0.42}, parameters);
  EXPECT_THAT(result.Softmax(),
              ::testing::ElementsAre(::testing::FloatNear(0.15, .001f),
                                     ::testing::FloatNear(0.396, .001f),
                                     ::testing::FloatNear(0.452, .001f)));
  EXPECT_NEAR(loss_fn.Loss(result, "Setosa"), 1.89, 0.01);
  EXPECT_THAT(loss_fn.Gradient(result, "Setosa"),
              ::testing::ElementsAre(::testing::FloatNear(-0.849, .001f),
                                     ::testing::FloatNear(0.396, .001f),
                                     ::testing::FloatNear(0.452, .001f)));

  result = model({0.5, 0.37}, parameters);
  EXPECT_THAT(result.Softmax(),
              ::testing::ElementsAre(::testing::FloatNear(0.04, .001f),
                                     ::testing::FloatNear(0.653, .001f),
                                     ::testing::FloatNear(0.306, .001f)));
  EXPECT_NEAR(loss_fn.Loss(result, "Versicolor"), 0.425, 0.01f);
  EXPECT_THAT(loss_fn.Gradient(result, "Versicolor"),
              ::testing::ElementsAre(::testing::FloatNear(0.04, .001f),
                                     ::testing::FloatNear(-0.347, .001f),
                                     ::testing::FloatNear(0.305, .001f)));

  result = model({1, 0.54}, parameters);
  EXPECT_THAT(result.Softmax(),
              ::testing::ElementsAre(::testing::FloatNear(0.035119, .001f),
                                     ::testing::FloatNear(0.259496, .001f),
                                     ::testing::FloatNear(0.705384, .001f)));
  EXPECT_NEAR(loss_fn.Loss(result, "Virginica"), 0.349, 0.01f);
  EXPECT_THAT(loss_fn.Gradient(result, "Virginica"),
              ::testing::ElementsAre(::testing::FloatNear(0.035, .001f),
                                     ::testing::FloatNear(0.259, .001f),
                                     ::testing::FloatNear(-0.295, .001f)));
}

}  // namespace
}  // namespace uchen::training::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}