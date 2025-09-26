#include "uchen/math/matrix.h"

#include <sys/stat.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"  // IWYU pragma: keep
#include "absl/strings/str_cat.h"

#include "gmock/gmock.h"

namespace uchen::math::testing {

namespace {

TEST(MatrixTest, DefaultConstructor) {
  RowMajorMatrix<3, 4> m(
      std::array<float, 12>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  EXPECT_THAT(m, ::testing::ElementsAre(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
}

TEST(MatrixTest, StrCat) {
  std::array<float, 80> data;
  data.fill(1);
  RowMajorMatrix<20, 4> m(std::move(data));
  static_assert(std::random_access_iterator<decltype(m.begin())>);
  EXPECT_TRUE(std::random_access_iterator<decltype(m.begin())>);
  std::string s = absl::StrCat(m);
  EXPECT_EQ(s, "(20,4){1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,...}");
}

TEST(FnMatrixTest, Basics) {
  auto m = FnMatrix<3, 2>(
      +[](uint32_t r, uint32_t c) { return static_cast<float>(r * 10 + c); });
  static_assert(std::bidirectional_iterator<decltype(m.begin())>);
  EXPECT_EQ(m.GetRowMajor(0), 0);
  EXPECT_EQ(m.GetRowMajor(1), 1);
  EXPECT_EQ(m.GetRowMajor(2), 10);
  EXPECT_EQ(m.GetRowMajor(5), 21);
  EXPECT_THAT(m, ::testing::ElementsAre(0, 1, 10, 11, 20, 21));
  EXPECT_EQ(absl::StrCat(m), "(3,2){0,1,10,11,20,21}");
}

TEST(MatrixTest, AssignRowMatrix) {
  RowMajorMatrix<2, 3> a(std::array<float, 6>{1, 2, 3, 4, 5, 6});
  RowMajorMatrix<2, 3> b = a;
  EXPECT_THAT(b, ::testing::ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(MatrixTest, AssignFnMatrix) {
  RowMajorMatrix<2, 3> b = FnMatrix<2, 3>(
      +[](uint32_t r, uint32_t c) { return static_cast<float>(r * 10 + c); });
  EXPECT_THAT(b, ::testing::ElementsAre(0, 1, 2, 10, 11, 12));
}

TEST(MatrixTest, AssignOps) {
  RowMajorMatrix<2, 3> a(std::array<float, 6>{1, 2, 3, 4, 5, 6});
  RowMajorMatrix<2, 3> b(std::array<float, 6>{10, 20, 30, 40, 50, 60});
  a += b;
  EXPECT_THAT(a, ::testing::ElementsAre(11, 22, 33, 44, 55, 66));
  a -= FnMatrix<2, 3>(
      +[](uint32_t r, uint32_t c) { return static_cast<float>(r * 10 + c); });
  EXPECT_THAT(a, ::testing::ElementsAre(11, 21, 31, 34, 44, 54));
  a *= 4;
  EXPECT_THAT(a, ::testing::ElementsAre(44, 84, 124, 136, 176, 216));
  a /= 2;
  EXPECT_THAT(a, ::testing::ElementsAre(22, 42, 62, 68, 88, 108));
}

TEST(MatrixTest, AssignToIterator) {
  RowMajorMatrix<3, 2> a(std::array<float, 6>{1, 2, 3, 4, 5, 6});
  *a.begin() = 10;
  EXPECT_THAT(a, ::testing::ElementsAre(10, 2, 3, 4, 5, 6));
}

TEST(MatrixTest, RowMajorTwoArg) {
  auto rm = FnMatrix<3, 3>(
      +[](uint32_t r, uint32_t c) { return static_cast<float>(r * 10 + c); });
  EXPECT_THAT(rm, ::testing::ElementsAre(0, 1, 2, 10, 11, 12, 20, 21, 22));
  std::array<float, 9> data;
  AsRowMajorView<3, 3>(std::span<float>(data)) = rm;
  EXPECT_THAT(data, ::testing::ElementsAre(0, 1, 2, 10, 11, 12, 20, 21, 22));
  AsColumnMajorView<3, 3>(std::span<float>(data)) = rm;
  EXPECT_THAT(data, ::testing::ElementsAre(0, 10, 20, 1, 11, 21, 2, 12, 22));
  auto cm = FnMatrix<3, 3>(
      +[](uint32_t r, uint32_t c) { return static_cast<float>(r * 10 + c); });
  AsRowMajorView<3, 3>(std::span<float>(data)) = cm;
  EXPECT_THAT(data, ::testing::ElementsAre(0, 1, 2, 10, 11, 12, 20, 21, 22));
  AsColumnMajorView<3, 3>(std::span<float>(data)) = cm;
  EXPECT_THAT(data, ::testing::ElementsAre(0, 10, 20, 1, 11, 21, 2, 12, 22));
}

TEST(MatrixTest, IterateAcrossRowMajor) {
  RowMajorMatrix<3, 2> m =
      FnMatrix<3, 2>(+[](size_t a) { return static_cast<float>(a); });
  EXPECT_FLOAT_EQ(m.GetColumnMajor(1), 2);
  m.GetColumnMajorRef(2) = 42;
  EXPECT_FLOAT_EQ(m.GetRowMajor(4), 42);
  *++m.begin_across() = 10;
  std::vector<float> v(m.begin_across(), m.end_across());
  EXPECT_THAT(v, ::testing::ElementsAre(0, 10, 42, 1, 3, 5));
}

TEST(MatrixTest, IterateAcrossColumnMajor) {
  ColumnMajorMatrix<3, 2> m =
      FnMatrix<3, 2>(+[](size_t a) { return static_cast<float>(a); });
  EXPECT_FLOAT_EQ(m.GetRowMajor(2), 2);
  m.GetRowMajorRef(2) = 42;
  EXPECT_THAT(std::vector<float>(m.begin_across(), m.end_across()),
              ::testing::ElementsAre(0, 42, 4, 1, 3, 5));
  auto it = m.begin_across();
  ++it;
  *it = 10;
  std::vector<float> v(m.begin_across(), m.end_across());
  EXPECT_THAT(v, ::testing::ElementsAre(0, 10, 4, 1, 3, 5));
}

TEST(MatrixTest, Transposed) {
  RowMajorMatrix<3, 2> m =
      FnMatrix<3, 2>(+[](size_t a) { return static_cast<float>(a); });
  auto t = m.transposed();
  EXPECT_THAT(t, ::testing::ElementsAre(0, 2, 4, 1, 3, 5));
}

}  // namespace

}  // namespace uchen::math::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}