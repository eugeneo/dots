#include "uchen/text/attention.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "uchen/math/matrix.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/text/embeddings.h"

namespace std {

template <size_t S>
std::ostream& operator<<(std::ostream& os, const std::span<const float, S>& d) {
  bool first = true;
  for (const auto a : d) {
    if (!first) {
      os << ", ";
    } else {
      first = false;
    }
    os << a;
  }
  return os;
}

template <size_t S>
std::ostream& operator<<(std::ostream& os, const std::array<float, S>& d) {
  return os << std::span<const float, S>(d);
}

}  // namespace std

namespace uchen::text::testing {

template <size_t Rows, size_t Cols>
std::array<float, Rows * Cols> Pack(
    std::initializer_list<std::array<float, Cols>> weights) {
  std::array<float, Rows * Cols> packed;
  auto it = packed.begin();
  for (const auto& row : weights) {
    std::copy(row.begin(), row.end(), it);
    it += Cols;
  }
  return packed;
}

template <size_t Rows, size_t Cols>
std::array<float, Rows*(Cols + 1)> Pack(
    std::array<float, Rows> biases,
    std::initializer_list<std::array<float, Cols>> weights) {
  std::array<float, Rows*(Cols + 1)> packed;
  std::copy(biases.begin(), biases.end(), packed.begin());
  auto packed_weights = Pack<Rows, Cols>(weights);
  std::copy(packed_weights.begin(), packed_weights.end(),
            packed.begin() + Rows);
  return packed;
}

template <size_t Rows, size_t Cols>
std::array<float, Rows*(Cols)> PackT(
    std::initializer_list<std::array<float, Cols>> weights) {
  std::array<float, Rows*(Cols)> packed;
  size_t col = 0;
  for (const auto& weights_row : weights) {
    for (size_t j = 0; j < weights_row.size(); j++) {
      packed[j * Cols + col] = weights_row[j];
    }
    col += 1;
  }
  return packed;
}

template <typename... Args>
auto JoinArrays(Args... args) {
  std::array<float, (std::tuple_size_v<Args> + ...)> result;
  auto it = result.begin();
  for (const auto& a : {args...}) {
    std::copy(a.begin(), a.end(), it);
    it += a.size();
  }
  return result;
}

constexpr std::array<float, 45> kParameters = {
    // Embeddings params
    -1.f,
    -1.f,
    -1.f,
    1.f,
    1.f,
    1.f,
    2.f,
    2.f,
    2.f,
    // Attention params
    // Keys matrix
    .1f,
    .2f,
    .3f,
    .4f,
    .5f,
    .6f,
    .7f,
    .8f,
    .9f,
    // Queries matrix
    .1f,
    .2f,
    .3f,
    .4f,
    .5f,
    .6f,
    .7f,
    .8f,
    .9f,
    // Values matrix
    .1f,
    .2f,
    .3f,
    .4f,
    .5f,
    .6f,
    .7f,
    .8f,
    .9f,
};

TEST(AttentionTest, StacksOnEmbeddings) {
  constexpr auto model = Embeddings<2, 4, 3> | Attention;
  ModelParameters p(&model, kParameters);
  std::array<uint32_t, 2> data = {2, 1};
  auto res = model(data, p);
  EXPECT_THAT(res, ::testing::ElementsAre(::testing::FloatNear(4.798, .001),
                                          ::testing::FloatNear(4.79996, .001),
                                          ::testing::FloatNear(4.6669, .001),
                                          ::testing::FloatNear(2.39894, .001),
                                          ::testing::FloatNear(2.39998, .001),
                                          ::testing::FloatNear(2.33345, .001)));
}

TEST(AttentionLayerContextTest, ValuesCalculation) {
  impl::AttentionLayerContext<3, 2> context;
  std::array parameters = JoinArrays(
      PackT<2, 2>({{.8, .9}, {1.1, 1.2}}), PackT<2, 2>({{2, 4}, {6, 8}}),
      PackT<2, 2>({{.5, .6}, {1.5, 1.6}}), std::array<float, 4>{0},
      std::array<float, 4>{0}, std::array<float, 4>{0},
      std::array<float, 4>{0});
  std::array input = Pack<3, 2>({{.1, .2}, {1.3, 1.4}, {.5, .6}});
  auto result =
      context.Calculate(math::RowMajorView<3, 2>(input), {parameters, nullptr});
  EXPECT_THAT(context.keys(), ::testing::ElementsAre(.3, .33, 2.58, 2.85,
                                                     ::testing::FloatEq(1.06),
                                                     ::testing::FloatEq(1.17)));
  EXPECT_THAT(context.queries(),
              ::testing::ElementsAre(
                  ::testing::FloatEq(1.4), ::testing::FloatEq(2), 11, 16.4,
                  ::testing::FloatEq(4.6), ::testing::FloatEq(6.8)));
  EXPECT_THAT(context.values(),
              ::testing::ElementsAre(
                  ::testing::FloatEq(.35), ::testing::FloatEq(.38), 2.75, 3.02,
                  ::testing::FloatEq(1.15), ::testing::FloatEq(1.26)));
  EXPECT_THAT(context.attention(),
              ::testing::ElementsAre(33.676, ::testing::FloatEq(50.12),
                                     ::testing::FloatEq(37.194), 55.356));
  math::ColumnMajorMatrix<2, 2> as = context.attention_softmax();
  EXPECT_THAT(as, ::testing::ElementsAre(::testing::FloatNear(0.029, 0.001),
                                         ::testing::FloatNear(0.005, 0.001),
                                         ::testing::FloatNear(0.97, 0.01),
                                         ::testing::FloatNear(0.995, 0.001)));
  EXPECT_EQ(as.GetRowMajor(0) + as.GetRowMajor(2), 1);
  EXPECT_EQ(as.GetRowMajor(1) + as.GetRowMajor(3), 1);
  EXPECT_THAT(result,
              ::testing::ElementsAre(
                  ::testing::FloatNear(.35 * .029 + .38 * 0.97, .01),
                  ::testing::FloatNear(.35 * 0.005 + .38 * 0.99, .01),
                  ::testing::FloatNear(2.75 * .029 + 3.02 * 0.97, .01),
                  ::testing::FloatNear(2.75 * .005 + 3.02 * 0.995, .01),
                  ::testing::FloatNear(1.15 * .029 + 1.26 * 0.97, .01),
                  ::testing::FloatNear(1.15 * .005 + 1.26 * 0.99, .01)));
}

}  // namespace uchen::text::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}