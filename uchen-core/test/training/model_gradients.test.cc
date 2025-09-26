#include "uchen/training/model_gradients.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/parameters.h"
#include "uchen/training/loss.h"
#include "uchen/vector.h"

namespace uchen::training::testing {

namespace {

TEST(ModelGradientsTest, Basic) {
  Model model =
      Model<Linear<Vector<float, 1>, 2>>() | layers::Relu | layers::Linear<1>;
  Vector<float, 1> input(2);
  uchen::training::ForwardPassResult forward_pass(
      &model, input, {&model, {1, -1, 1, -1, -3, 3, -3}});
  EXPECT_THAT(forward_pass.result(), ::testing::ElementsAre(6));
  EXPECT_THAT(forward_pass.CalculateParameterGradients({-1}).second,
              ::testing::ElementsAre(-3, 0, -6, 0, -1, -3, 0));
}

TEST(ModelGradientsTest, GradientCalculator) {
  Model m = layers::Input<Vector<float, 2>> | layers::Linear<3>;
  ModelParameters parameters(&m, 1);
  EXPECT_THAT(m({.5, 1.5}, parameters), ::testing::ElementsAre(3, 3, 3));
  SquaredLoss<float, 3> loss_fn;
  Vector<float, 2> input = {.5f, 1.5f};
  ForwardPassResult fp(&m, input, parameters);
  EXPECT_THAT(fp.result(), ::testing::ElementsAre(3, 3, 3));
  EXPECT_NEAR(loss_fn.Loss(fp.result(), {2, 3, 4}), 0.666, 0.001);
  Vector loss_gradients = loss_fn.Gradient(fp.result(), {2, 3, 4});
  auto gradients = fp.CalculateParameterGradients(loss_gradients).second;
  EXPECT_THAT(std::vector({gradients[1], gradients[4], gradients[7]}),
              ::testing::ElementsAre(0, 0, 0));
}

}  // namespace
}  // namespace uchen::training::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}