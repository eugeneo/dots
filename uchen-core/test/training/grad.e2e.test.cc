#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "uchen/layers.h"
#include "uchen/parameters.h"
#include "uchen/training/training.h"
#include "uchen/vector.h"

namespace uchen::training::testing {

namespace {

template <typename T>
auto GenerateLinearSeq(T start, T end, T step, T k, T b) {
  std::vector<std::pair<Vector<T, 1>, Vector<T, 1>>> result;
  for (T i = start; i < end; i++) {
    auto y = k * (i * step) + b;
    result.emplace_back(Vector<T, 1>{i}, Vector<T, 1>{y});
  }
  return result;
}

TEST(GradientDescentTest, LinearRegression) {
  auto model = layers::Input<Vector<float, 1>> | layers::Linear<2> |
               layers::Relu | layers::Linear<1>;
  auto inputs = GenerateLinearSeq<float>(-10, 0, 1, -0.5, 0);
  auto second_half = GenerateLinearSeq<float>(0, 10, 1, 1.5, 0);
  std::copy(second_half.begin(), second_half.end(), std::back_inserter(inputs));
  EXPECT_EQ(model.all_parameters_count(), 7);
  std::vector validation = {inputs.front(), inputs.back()};
  Training training(&model, {&model, {0, 0, -5, 5, 0, 0, 0}});
  TrainingData<Vector<float, 1>, Vector<float, 1>> data(inputs);
  for (size_t generation = 1; training.Loss(data) > 0.0001; ++generation) {
    training = training.Generation(data, 0.001);
    ASSERT_LT(generation, 500);
  }
  ModelParameters parameters = training.parameters();
  EXPECT_NEAR(model({-20}, parameters)[0], 10, 0.15);
  EXPECT_NEAR(model({0}, parameters)[0], 0, 0.2);
  EXPECT_NEAR(model({20}, parameters)[0], 30, 1.1);
}

TEST(GradientDescentTest, TwoXPlusOne) {
  Model model = layers::Input<Vector<float, 1>> | layers::Linear<1>;
  TrainingData<Vector<float, 1>, Vector<float, 1>> training = {
      {{-2}, {-3}},
      {{2}, {5}},
  };
  TrainingData<Vector<float, 1>, Vector<float, 1>> validation = {
      {{{5}, {11}}},
  };
  Training t(&model, {&model, {0, -20}});
  for (size_t generation = 1; t.Loss(validation) > 0.0001; ++generation) {
    ASSERT_LT(generation, 500);
    generation += 1;
    t = t.Generation(training, .1);
  }
  ModelParameters parameter = t.parameters();
  EXPECT_NEAR(model({7}, parameter)[0], 15, 0.08);
  EXPECT_NEAR(model({10}, parameter)[0], 21, 0.08);
}

}  // namespace
}  // namespace uchen::training::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}