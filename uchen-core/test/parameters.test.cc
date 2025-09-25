#include "uchen/parameters.h"

#include <array>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "uchen/layers.h"
#include "uchen/linear.h"

namespace uchen::testing {

TEST(ModelParameters, IsIterable) {
  Model m = layers::Input<Vector<float, 10>> | layers::Linear<10> |
            layers::Linear<10>;
  ModelParameters parameters(&m);
  static_assert(std::random_access_iterator<
                internal::ModelParametersIterator<decltype(m)>>);
  EXPECT_THAT(parameters,
              ::testing::SizeIs(decltype(m)::all_parameters_count()));
}

TEST(ParametersTest, ZeroOffset) {
  std::array<float, 8> store = {1, 2, 3, 4, 5, 6, 7, 8};
  Parameters<6> p(store);
  EXPECT_THAT(p, ::testing::SizeIs(6));
  EXPECT_EQ(p[0], 1);
  EXPECT_EQ(p[3], 4);
  EXPECT_EQ(p.sum(), 1 + 2 + 3 + 4 + 5 + 6);
}

TEST(ParametersTest, NonZeroOffset) {
  std::array<float, 8> store = {1, 2, 3, 4, 5, 6, 7, 8};
  Parameters<6> p1(store);
  auto p = p1.starting<2>();
  EXPECT_THAT(p, ::testing::ElementsAre(3, 4, 5, 6));
  EXPECT_EQ(p[0], 3);
  EXPECT_EQ(p[3], 6);
  EXPECT_EQ(p.sum(), 3 + 4 + 5 + 6);
  EXPECT_THAT((p.starting<1, 1>()), ::testing::ElementsAre(4));
  EXPECT_EQ((p.starting<1, 1>()).sum(), 4);
}

TEST(ParametersTest, LayerForIndexSimpleTest) {
  Model<Linear<Vector<float, 2>, 1>> m;
  using M = decltype(m);
  std::vector<size_t> calculated;
  for (size_t i = 0; i < M::all_parameters_count(); ++i) {
    calculated.emplace_back(internal::LayerIndexes<M>::layer_for_index(i));
  }
  EXPECT_THAT(calculated, ::testing::ElementsAre(0, 0, 0));
  EXPECT_EQ(internal::LayerIndexes<M>::layer_for_index(1000), 1);
}

TEST(ParametersTest, LayerForIndexEvenCount) {
  Model<Linear<Vector<float, 2>, 2>, Linear<Vector<float, 2>, 2>> m;
  using M = decltype(m);
  std::vector<size_t> calculated;
  for (size_t i = 0; i < M::all_parameters_count(); ++i) {
    calculated.emplace_back(internal::LayerIndexes<M>::layer_for_index(i));
  }
  EXPECT_THAT(calculated,
              ::testing::ElementsAre(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1));
  EXPECT_EQ(internal::LayerIndexes<M>::layer_for_index(1000), 2);
}

TEST(ParametersTest, LayerForIndexFullTest) {
  Model m = layers::Input<Vector<float, 1>> | layers::Relu | layers::Relu |
            layers::Linear<2> | layers::Relu | layers::Linear<2>;
  using M = decltype(m);
  std::vector<size_t> calculated;
  for (size_t i = 0; i < M::all_parameters_count(); ++i) {
    calculated.emplace_back(internal::LayerIndexes<M>::layer_for_index(i));
  }
  EXPECT_THAT(calculated, ::testing::ElementsAre(3, 3, 3, 3, 5, 5, 5, 5, 5, 5));
  EXPECT_EQ(internal::LayerIndexes<M>::layer_for_index(1000), M::kLayers);
}

}  // namespace uchen::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}