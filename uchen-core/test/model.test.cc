#include "uchen/model.h"

#include <cstddef>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/test_lib.h"
#include "uchen/layer_traits.h"
#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/memory.h"
#include "uchen/parameters.h"
#include "uchen/vector.h"

namespace uchen {

class TestInputLayer {
 public:
  using input_t = int;
  using output_t = float;

  float operator()(int input) const { return input * 2.0f; }
};

template <size_t ParCount>
class TestLayer {
 public:
  constexpr static size_t parameter_count = ParCount;
  using input_t = float;
  using output_t = float;

  float operator()(float input, const Parameters<ParCount>& parameters) const {
    return input + parameters.sum();
  }
};

template <size_t ParCount>
auto ParameterProvider(const TestLayer<ParCount>& layer,
                       std::span<const float> data,
                       std::shared_ptr<memory::Deletable> handle) {
  return Parameters<ParCount>(data, std::move(handle));
}

template <size_t ParCount>
class TestLayerDesc {
 public:
  template <concepts::InputLayer IL>
  using stack_t = TestLayer<IL::parameter_count + ParCount>;

  template <typename Model>
  constexpr auto stack(const Model& model) const {
    return TestLayer<Model::all_parameters_count() + ParCount>();
  }
};

template <>
struct LayerTraits<TestInputLayer, int> : public LayerTraitFields<float> {};

template <size_t S>
struct LayerTraits<TestLayer<S>, float> : public LayerTraitFields<float, S> {};

constexpr Model<TestInputLayer> kTestModel;
template <size_t ParCount>
constexpr Layer<TestLayerDesc<ParCount>> kTestLayer;

TEST(ModelTest, SingleLayer) {
  Model<TestInputLayer> model;
  EXPECT_EQ(model.kLayers, 1);
  EXPECT_EQ(model(5, ModelParameters(&model, {})), 10.f);
}

TEST(ModelTest, TwoLayers) {
  Model<TestInputLayer, TestLayer<3>> model(
      std::make_tuple(TestInputLayer(), TestLayer<3>()));
  EXPECT_EQ(model.kLayers, 2);
  EXPECT_EQ(decltype(model)::L<1>::parameter_count, 3);
}

TEST(ModelTest, FourLayers) {
  Model<TestInputLayer, TestLayer<3>, TestLayer<2>, TestLayer<4>> model =
      Model(std::make_tuple(TestInputLayer(), TestLayer<3>(), TestLayer<2>(),
                            TestLayer<4>()));
  EXPECT_EQ(model.kLayers, 4);
  EXPECT_EQ(model.all_parameters_count(), 9);
  EXPECT_EQ(decltype(model)::L<1>::parameter_count, 3);
  EXPECT_EQ(decltype(model)::L<3>::parameter_count, 4);
}

TEST(ModelTest, Piping) {
  auto model = kTestModel | kTestLayer<3> | kTestLayer<2> | kTestLayer<4>;
  EXPECT_EQ(model.kLayers, 4);
  EXPECT_EQ(model.all_parameters_count(), 20);
  EXPECT_EQ(model(5, {&model, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20}}),
            220);
}

TEST(ModelTest, PipingLayers) {
  auto layers = kTestLayer<1> | kTestLayer<2>;
  auto model = kTestModel | layers | layers;
  EXPECT_EQ(model.kLayers, 5);
  EXPECT_EQ(model.all_parameters_count(), 20);
  EXPECT_EQ(model(5, {&model, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20}}),
            220);
}

TEST(ModelTest, Linear) {
  Model m1 = layers::Input<Vector<float, 2>> | layers::Linear<2>;
  EXPECT_THAT(m1({1, 2}, ModelParameters(&m1, testing::RearrangeLinear(
                                                  {{1, 2, 3}, {4, 5, 6}}))),
              ::testing::ElementsAre(8, 20));
  Model m3 = layers::Input<Vector<float, 2>> | layers::Linear<2> | layers::Relu;
  EXPECT_THAT(m3({1, 2}, ModelParameters(&m3, testing::RearrangeLinear(
                                                  {{-1, -2, -3}, {4, 5, 6}}))),
              ::testing::ElementsAre(0, 20));
}

}  // namespace uchen

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}