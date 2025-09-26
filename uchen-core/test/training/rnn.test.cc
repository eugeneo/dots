#include "uchen/training/rnn.h"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <valarray>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "test/test_lib.h"
#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/parameters.h"
#include "uchen/rnn.h"
#include "uchen/training/backprop.h"
#include "uchen/training/model_gradients.h"
#include "uchen/vector.h"

namespace uchen::training {

namespace testing {

namespace {

struct TestInput {
  size_t ind = 2;

 public:
  using value_type = Vector<float, 1>;
  class Iterator {};

  std::optional<Vector<float, 1>> Next() {
    if (--ind > 0) {
      return std::make_optional<Vector<float, 1>>(static_cast<float>(ind));
    }
    return std::nullopt;
  }
};

TEST(RnnTrainingTest, SmallGradient) {
  Model m =
      layers::Rnn<std::span<const Vector<float, 1>>, 1>(layers::Linear<1>);
  std::array input = {Vector<float, 1>{.5}, Vector<float, 1>{-.5}};
  std::span<const Vector<float, 1>> s = input;
  ModelParameters parameters(&m, std::valarray{0.f, 1.f, .2f, 0.f, 2.f});
  ForwardPassResult fp(&m, s, parameters);
  EXPECT_THAT(fp.result(), ::testing::ElementsAre(-.3));
  EXPECT_THAT(fp.CalculateParameterGradients({1}).second,
              ::testing::ElementsAre(1.4, -0.3, 1, .2, 0.1));
}

TEST(RnnTrainingTest, NewImplementation) {
  Model m =
      layers::Rnn<std::span<const Vector<float, 1>>, 2>(layers::Linear<1>);
  std::array data = {Vector<float, 1>{2}, Vector<float, 1>{1}};
  EXPECT_THAT(m(data, ModelParameters(&m, 1)), ::testing::ElementsAre(10));
  ModelParameters parameters(&m, 1);
  std::span<const Vector<float, 1>> span = data;
  ForwardPassResult forward_pass(&m, span, parameters);
  EXPECT_THAT(forward_pass.result(), ::testing::ElementsAre(10));
  auto parameter_gradients =
      forward_pass.CalculateParameterGradients({2}).second;
  EXPECT_THAT(parameter_gradients,
              ::testing::ElementsAre(6, 10, 8, 8, 2, 2, 6, 6));
}

}  // namespace
}  // namespace testing
}  // namespace uchen::training

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}