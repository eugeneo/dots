#include "uchen/math/primitives.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/check.h"  // IWYU pragma: keep
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"  // IWYU pragma: keep

namespace uchen::math::testing {
namespace {

constexpr uint8_t kMaxLanes = 32;  // Future proofing!

}  // namespace

TEST(PrimitivesTest, DotProduct) {
  std::vector<float> a(31, 1.0f);
  std::vector<float> b(31, 2.0f);
  float sum = DotProduct(a, b);
  EXPECT_FLOAT_EQ(sum, 62.f);
}

TEST(PrimitivesTest, DotProduct_2LengthNoCrash) {
  std::vector<float> a({1.f, 3.f});
  std::vector<float> b({2.f, 4.f});
  float sum = DotProduct(a, b);
  EXPECT_FLOAT_EQ(sum, 14.f);
}

TEST(PrimitivesTest, DotProduct_MulLarge) {
  // Make it not a power of 2. Some SIMD pain.
  constexpr uint32_t N = 1024 * 512 - 3;
  std::vector<float> a(N);
  std::vector<float> b(N);
  for (size_t i = 0; i < N; ++i) {
    a[i] = 1.f / static_cast<float>(i + 2);
    b[i] = static_cast<float>(i + 2) / static_cast<float>(N);
  }
  float sum = DotProduct(a, b);
  EXPECT_NEAR(sum, 1, 0.0001);
}

TEST(PrimitivesTest, ColumnsByRow) {
  std::array<float, 64> a;
  std::array<float, 8> b = {1.f, -2.f, 3.f, -4.f, 5.f, -6.f, 7.f, -8.f};
  for (size_t c = 0; c < 8; ++c) {
    for (size_t r = 0; r < 8; ++r) {
      a[c * 8 + r] = static_cast<float>(r / std::abs(b[c]));
    }
  }
  std::array<float, 8> out;
  out.fill(0.f);
  MatrixByVector(a, b, out);
  EXPECT_THAT(out, ::testing::Each(::testing::FloatEq(0.f)));
}

TEST(PrimitivesTest, ColumnsByRow2x1) {
  std::array<float, 2> a = {1.f, 2.f};
  std::array<float, 1> b = {3.f};
  std::array<float, 2> out;
  out.fill(0.f);
  MatrixByVector(a, b, out);
  EXPECT_THAT(out,
              ::testing::ElementsAre(3.f, ::testing::FloatNear(6.f, 0.00001f)));
}

TEST(PrimitivesTest, MatrixByVectorBulk) {
  const size_t Lanes = GetLanesForTest() * 2;
  CHECK_LE(Lanes, kMaxLanes);
  std::array<float, kMaxLanes * kMaxLanes * 4> a;
  std::array<float, kMaxLanes * 2> b;
  for (size_t i = 0; i < Lanes; ++i) {
    b[i] = static_cast<float>(i + 1) * (i & 1 ? 1.f : -1.f);
  }
  for (size_t c = 0; c < Lanes; ++c) {
    for (size_t r = 0; r < Lanes; ++r) {
      a[c * Lanes + r] = static_cast<float>(r / std::abs(b[c]));
    }
  }
  std::array<float, kMaxLanes * 2> out;
  out.fill(0.f);
  MatrixByVector(std::span(a).first(Lanes * Lanes), std::span(b).first(Lanes),
                 std::span(out).first(Lanes));
  EXPECT_THAT(out, ::testing::Each(::testing::FloatNear(0.f, .1e-4)));
}

TEST(PrimitivesTest, MatrixByVectorBottom) {
  size_t Cols = GetLanesForTest() * 2;
  size_t Rows = GetLanesForTest() - 1;
  LOG(INFO) << "Cols: " << Cols << " Rows: " << Rows;
  std::array<float, kMaxLanes * kMaxLanes * 2> a;
  std::array<float, kMaxLanes * 2> b;
  for (size_t i = 0; i < Cols; ++i) {
    b[i] = static_cast<float>(i + 1) * ((i & 1) == 0 ? 1.f : -1.f);
  }
  for (size_t c = 0; c < Cols; ++c) {
    for (size_t r = 0; r < Rows; ++r) {
      a[c * Rows + r] = static_cast<float>(r / std::abs(b[c]));
    }
  }
  std::array<float, kMaxLanes * 2> out;
  out.fill(0.f);
  MatrixByVector(std::span(a).first(Cols * Rows), std::span(b).first(Cols),
                 std::span(out).first(Rows));
  EXPECT_THAT(out, ::testing::Each(::testing::FloatNear(0.f, .1e-4f)));
}

TEST(PrimitivesTest, MatrixByVectorRight) {
  size_t Cols = GetLanesForTest() - 1;
  size_t Rows = GetLanesForTest() * 2;
  LOG(INFO) << "Cols: " << Cols << " Rows: " << Rows;
  std::array<float, kMaxLanes * kMaxLanes * 2> a;
  std::array<float, kMaxLanes * 2> b;
  for (size_t i = 0; i < Cols; ++i) {
    b[i] = static_cast<float>(i + 1) * ((i & 1) == 0 ? 1.f : -1.f);
  }
  for (size_t c = 0; c < Cols; ++c) {
    for (size_t r = 0; r < Rows; ++r) {
      a[c * Rows + r] = static_cast<float>(r / std::abs(b[c]));
    }
  }
  std::array<float, kMaxLanes * 2> out;
  out.fill(0.f);
  MatrixByVector(std::span(a).first(Cols * Rows), std::span(b).first(Cols),
                 std::span(out).first(Rows));
  for (float i = 0; i < Rows; ++i) {
    EXPECT_NEAR(out[i], i, 0.0001f);
  }
}

TEST(PrimitivesTest, MatrixByVectorCorner) {
  size_t Cols = GetLanesForTest() - 1;
  size_t Rows = GetLanesForTest() - 1;
  std::array<float, kMaxLanes * kMaxLanes * 2> a;
  std::array<float, kMaxLanes * 2> b;
  for (size_t i = 0; i < Cols; ++i) {
    b[i] = static_cast<float>(i + 1) * ((i & 1) == 0 ? 1.f : -1.f);
  }
  for (size_t c = 0; c < Cols; ++c) {
    for (size_t r = 0; r < Rows; ++r) {
      a[c * Rows + r] = static_cast<float>(r / std::abs(b[c]));
    }
  }
  std::array<float, kMaxLanes * 2> out;
  out.fill(0.f);
  MatrixByVector(std::span(a).first(Cols * Rows), std::span(b).first(Cols),
                 std::span(out).first(Rows));
  for (float i = 0; i < Rows; ++i) {
    EXPECT_NEAR(out[i], i, 0.0001f);
  }
}

TEST(PrimitivesTest, MatrixByVectorAll) {
  size_t Cols = GetLanesForTest() * 2 - 1;
  size_t Rows = GetLanesForTest() * 2 - 1;
  LOG(INFO) << "Cols: " << Cols << " Rows: " << Rows;
  std::array<float, kMaxLanes * kMaxLanes * 2> a;
  std::array<float, kMaxLanes * 2> b;
  for (size_t i = 0; i < Cols; ++i) {
    b[i] = static_cast<float>(i + 1) * ((i & 1) == 0 ? 1.f : -1.f);
  }
  for (size_t c = 0; c < Cols; ++c) {
    for (size_t r = 0; r < Rows; ++r) {
      a[c * Rows + r] = static_cast<float>(r / std::abs(b[c]));
    }
  }
  std::array<float, kMaxLanes * 2> out;
  out.fill(0.f);
  MatrixByVector(std::span(a).first(Cols * Rows), std::span(b).first(Cols),
                 std::span(out).first(Rows));
  for (float i = 0; i < Rows; ++i) {
    EXPECT_NEAR(out[i], i, 0.0001f);
  }
}

}  // namespace uchen::math::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}