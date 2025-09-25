#include <cstddef>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "uchen/math/matrix.h"

namespace uchen::math::testing {
namespace {

TEST(MatrixMathTest, Add) {
  RowMajorMatrix<3, 2> a = FnMatrix<3, 2>(+[](size_t c) { return 1.f; });
  auto b = FnMatrix<3, 2>(+[](uint32_t r, uint32_t c) { return r * 2 + c; });
  RowMajorMatrix<3, 2> c = a + b;
  EXPECT_THAT(c, ::testing::ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(MatrixMathTest, Multiply) {
  Matrix a = FnMatrix<3, 2>(+[](size_t c) -> float { return c + 1; });
  Matrix b = FnMatrix<2, 2>(+[](uint32_t r, uint32_t c) {
    return static_cast<float>(r == c) * 0.1f;
  });
  RowMajorMatrix<3, 2> c = a * b;
  EXPECT_THAT(c, ::testing::ElementsAre(.1, .2, .3, .4, .5, .6));
}

TEST(MatrixMathTest, Linear) {
  Matrix x = FnMatrix<1, 3>(+[](size_t c) -> float { return c + 1; });
  Matrix w = FnMatrix<3, 2>(+[](size_t i) -> float { return i * 0.1f; });
  Matrix b = FnMatrix<1, 2>(+[](size_t i) -> float { return 2 - i; });
  RowMajorMatrix<1, 2> y = x * w + b;
  EXPECT_THAT(y, ::testing::ElementsAre(1.6 + 2, 2.2 + 1));
}

TEST(MatrixMathTest, ColumnMajorMultiply) {
  ColumnMajorMatrix<2, 2> a =
      FnMatrix<2, 2>(+[](size_t c) { return (c + 1.f) / 10.f; });
  EXPECT_THAT(a, ::testing::ElementsAre(.1f, .2f, .3f, .4f));
  ColumnMajorMatrix<2, 2> b =
      FnMatrix<2, 2>(+[](size_t c) { return (c + 5.f) / 10.f; });
  ColumnMajorMatrix<2, 2> c = a * b;
  EXPECT_THAT(c, ::testing::ElementsAre(
                     ::testing::FloatEq(.19f), ::testing::FloatEq(.22f),
                     ::testing::FloatEq(.43f), ::testing::FloatEq(.5f)));
}

TEST(MatrixMathTest, RCToR) {
  static constexpr uint32_t N = 256;
  RowMajorMatrix<N, N> a =
      FnMatrix<N, N>(+[](size_t i) { return static_cast<float>(i); });
  ColumnMajorMatrix<N, N> i = FnMatrix<N, N>(
      +[](uint32_t r, uint32_t c) { return r == c ? 1.f : 0.f; });
  RowMajorMatrix<N, N> c = a * i;
  EXPECT_EQ(c, a);
}

TEST(MatrixMathTest, RCToC) {
  static constexpr uint32_t N = 256;
  RowMajorMatrix<N, N> a =
      FnMatrix<N, N>(+[](size_t i) { return static_cast<float>(i); });
  ColumnMajorMatrix<N, N> i = FnMatrix<N, N>(
      +[](uint32_t r, uint32_t c) { return r == c ? 1.f : 0.f; });
  ColumnMajorMatrix<N, N> c = a * i;
  EXPECT_EQ(c, a);
}

TEST(MatrixMathTest, CRToC) {
  static constexpr uint32_t N = 127;
  ColumnMajorMatrix<N, N> a =
      FnMatrix<N, N>(+[](size_t i) { return static_cast<float>(i); });
  RowMajorMatrix<N, N> i = FnMatrix<N, N>(
      +[](uint32_t r, uint32_t c) { return r == c ? 1.f : 0.f; });
  ColumnMajorMatrix<N, N> c = a * i;
  for (size_t i = 0; i < c.columns() * c.rows(); ++i) {
    ASSERT_FLOAT_EQ(c.GetRowMajor(i), a.GetRowMajor(i)) << i;
  }
}

TEST(MatrixMathTest, CRToC1) {
  static constexpr uint32_t N = 16;
  ColumnMajorMatrix<1, N> a =
      FnMatrix<1, N>(+[](size_t i) { return static_cast<float>(i); });
  RowMajorMatrix<N, N> i = FnMatrix<N, N>(
      +[](uint32_t r, uint32_t c) { return r == c ? 1.f : 0.f; });
  ColumnMajorMatrix<1, N> c = a * i;
  for (size_t i = 0; i < c.columns() * c.rows(); ++i) {
    ASSERT_FLOAT_EQ(c.GetRowMajor(i), a.GetRowMajor(i)) << i;
  }
}

TEST(MatrixMathTest, CRToR1) {
  static constexpr uint32_t N = 127;
  ColumnMajorMatrix<N, N> a =
      FnMatrix<N, N>(+[](size_t i) { return static_cast<float>(i); });
  RowMajorMatrix<N, 1> i =
      FnMatrix<N, 1>(+[](size_t i) { return static_cast<float>(i == 1); });
  RowMajorMatrix<N, 1> c = a * i;
  for (size_t i = 0; i < N; ++i) {
    ASSERT_FLOAT_EQ(c.GetRowMajor(i), a.GetRowMajor(1 + (N * i))) << i;
  }
}

}  // namespace
}  // namespace uchen::math::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}