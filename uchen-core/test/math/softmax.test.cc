#include "uchen/math/softmax.h"

#include <array>
#include <cstddef>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gtest/gtest.h"
#include "uchen/math/concepts.h"
#include "uchen/math/matrix.h"

namespace uchen::math::testing {
namespace {

TEST(SoftmaxTest, SingleColumn) {
  auto m = FnMatrix<16, 1>(
      +[](uint32_t r, uint32_t c) { return 1.f + static_cast<float>(r == 4); });
  ColumnMajorMatrix<16, 1> s;
  Softmax(m, s);
  std::array<float, 16> expected;
  expected.fill(0.0564389);
  expected[4] = 0.153417;
  size_t i = 0;
  for (auto it1 = s.begin_across(), it2 = expected.begin();
       it1 != s.end_across(); ++it1, ++it2) {
    EXPECT_NEAR(*it1, *it2, 1e-3) << "element #" << i;
    ++i;
  }
}

class NameGenerator {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return absl::StrCat(name<std::tuple_element_t<0, T>>(), " x ",
                        name<std::tuple_element_t<1, T>>());
  }

 private:
  template <typename T>
  static std::string name() {
    if constexpr (RowContinuous<T>) {
      return "Row";
    } else if constexpr (ColumnContinuous<T>) {
      return "Column";
    } else {
      return "Unknown";
    }
  }
};

template <typename T>
class ThirtyOneByThirtyOne : public ::testing::Test {
 public:
  using input_type = std::tuple_element_t<0, T>;
  using output_type = std::tuple_element_t<1, T>;
};

using MyTypes31 = ::testing::Types<
    std::tuple<ColumnMajorMatrix<31, 31>, ColumnMajorMatrix<31, 31>>,
    std::tuple<RowMajorMatrix<31, 31>, RowMajorMatrix<31, 31>>,
    std::tuple<RowMajorMatrix<31, 31>, ColumnMajorMatrix<31, 31>>,
    std::tuple<ColumnMajorMatrix<31, 31>, RowMajorMatrix<31, 31>>>;

TYPED_TEST_SUITE(ThirtyOneByThirtyOne, MyTypes31, NameGenerator);

template <typename T>
class TwoByTwo : public ::testing::Test {
 public:
  using input_type = std::tuple_element_t<0, T>;
  using output_type = std::tuple_element_t<1, T>;
};

using MyTypes2 = ::testing::Types<
    std::tuple<ColumnMajorMatrix<2, 2>, ColumnMajorMatrix<2, 2>>,
    std::tuple<RowMajorMatrix<2, 2>, RowMajorMatrix<2, 2>>,
    std::tuple<RowMajorMatrix<2, 2>, ColumnMajorMatrix<2, 2>>,
    std::tuple<ColumnMajorMatrix<2, 2>, RowMajorMatrix<2, 2>>>;

TYPED_TEST_SUITE(TwoByTwo, MyTypes2, NameGenerator);

TYPED_TEST(TwoByTwo, Sm) {
  constexpr size_t N = 2;
  typename TestFixture::input_type m = FnMatrix<N, N>(
      +[](uint32_t r, uint32_t c) { return 1.f + static_cast<float>(r == c); });
  typename TestFixture::input_type s = Softmax(m);
  std::array<float, N * N> expected;
  expected.fill(0.2689f);
  for (size_t i = 0; i < N; ++i) {
    expected[i * N + i] = 0.7310f;
  }
  auto it = s.begin_across();
  for (size_t c = 0; c < N; ++c) {
    for (size_t r = 0; r < N; ++r) {
      EXPECT_NEAR(*it++, expected[c * N + r], 1e-3)
          << "element " << r << ", " << c;
    }
  }
}

TYPED_TEST(ThirtyOneByThirtyOne, Sm) {
  constexpr size_t N = TestFixture::input_type::R;
  typename TestFixture::input_type m = FnMatrix<N, N>(
      +[](uint32_t r, uint32_t c) { return 1.f + static_cast<float>(r == c); });
  typename TestFixture::input_type s = Softmax(m);
  std::array<float, N * N> expected;
  expected.fill(0.03f);
  for (size_t i = 0; i < N; ++i) {
    expected[i * N + i] = 0.083f;
  }
  auto it = s.begin_across();
  for (size_t c = 0; c < N; ++c) {
    for (size_t r = 0; r < N; ++r) {
      ASSERT_NEAR(*it++, expected[c * N + r], 1e-3)
          << "element " << r << ", " << c;
    }
  }
}

TEST(SimpleSoftmax, TakenFromAttnTest) {
  std::array<float, 6> m = {0, 1, 1, 0, .1, .1};
  math::RowMajorMatrix<2, 3> o = Softmax(math::AsColumnMajorView<2, 3>(m));
  EXPECT_THAT(o,
              ::testing::ElementsAre(::testing::FloatNear(0.269, 0.001),
                                     ::testing::FloatNear(0.731, 0.001), .5f,
                                     ::testing::FloatNear(0.731, 0.001),
                                     ::testing::FloatNear(0.269, 0.001), .5f));
}

}  // namespace
}  // namespace uchen::math::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}