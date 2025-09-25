#include "uchen/layers.h"

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/test_lib.h"
#include "uchen/linear.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/softmax.h"

namespace uchen::testing {

template <typename V, size_t Elements>
constexpr Model SimpleInput = uchen::layers::Input<Vector<V, Elements>>;

TEST(LayersTest, LinearThreeInputsOneOutput) {
  Linear<Vector<float, 3>, 1> model;
  std::array<float, 4> data = {4, -1, 2, -3};
  Parameters<4> params(data);
  std::array<float, 1> m;
  internal::InferenceLayerContext ctx(&m);
  EXPECT_THAT(model({1, -2, 3}, params, &ctx),
              ::testing::ElementsAre(-1 - 4 - 9 + 4));
}

TEST(LayersTest, LinearTwoInputsThreeOutputs) {
  Linear<Vector<float, 2>, 3> model;
  std::array<float, 9> parameters = {-3, -6, 9, -1, -4, 7, 2, 5, -8};
  std::array<float, 3> m;
  internal::InferenceLayerContext ctx(&m);
  EXPECT_THAT(model({1, -2}, Parameters<9>(parameters), &ctx),
              ::testing::ElementsAre(-1 - 4 - 3, -4 - 10 - 6, 7 + 16 + 9));
}

TEST(LayersTest, ReluTenElements) {
  auto result = ElementWise<Vector<float, 10>, Relu>()(
      Vector<float, 10>{-1, 2, -3, 4, -5, 6, -7, 8, -9, 10});
  EXPECT_THAT(result, ::testing::ElementsAre(0, 2, 0, 4, 0, 6, 0, 8, 0, 10));
}

TEST(LayersTest, Categories) {
  constexpr std::array<std::string_view, 3> aa = {"a", "b", "c"};
  constexpr Model model =
      SimpleInput<float, 2> | layers::Categories<std::string_view, 3>(aa);
  auto r = model(
      {1.f, 2.f},
      ModelParameters(&model,
                      RearrangeLinear({{1, .5, 0}, {.5, .5, 0}, {.5, 1, 0}})));
  EXPECT_EQ(r, "c");
  EXPECT_THAT(
      r.MatchDetails(),
      ::testing::ElementsAre(std::make_pair("a", 2), std::make_pair("b", 1.5),
                             std::make_pair("c", 2.5)));
}

TEST(LayersTest, SigmoidTest) {
  using Lim = std::numeric_limits<double>;

  Vector<double, 9> v = {
      Lim::min(),      -Lim::min(),      0, -0, Lim::max(), -Lim::max(),
      Lim::infinity(), -Lim::infinity(), 1,
  };
  Model m = layers::Input<decltype(v)> | layers::Sigmoid;
  EXPECT_THAT(m(v, ModelParameters<decltype(m)>(&m)),
              ::testing::ElementsAre(0.5, 0.5, 0.5, 0.5, 1, 0, 1, 0,
                                     ::testing::DoubleNear(0.731, 0.001)));
}

}  // namespace uchen::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}