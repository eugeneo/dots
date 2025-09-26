#include "uchen/tensor/float_tensor.h"

#include <array>
#include <cstddef>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"  // IWYU pragma: keep

#include "test/tensor/tensor_test_util.h"

namespace {

using uchen::core::FloatTensor;
using uchen::core::TensorProjection;
using uchen::core::testing::IotaTensor;

TEST(FloatTensorTest, Tensor3d) {
  FloatTensor t = IotaTensor<2, 4, 8>();
  std::array<float, t.kDims[0] * t.kDims[1] * t.kDims[2]> expected_data;
  std::iota(expected_data.begin(), expected_data.end(), 1.f);
  EXPECT_THAT(t.data(), ::testing::ElementsAreArray(expected_data));
}

TEST(FloatTensorTest, Projection) {
  FloatTensor t = IotaTensor<2, 8, 4>();
  TensorProjection projection = TensorProjection::dim_slice(t, 1, 4, 4);
  EXPECT_EQ(projection.rank(), 3);
  EXPECT_EQ(projection.dim(0), 2);
  EXPECT_EQ(projection.dim(1), 4);
  EXPECT_EQ(projection.dim(2), 4);
  EXPECT_EQ(projection(0, 0, 0), t(0, 4, 0));
  EXPECT_EQ(projection(0, 0, 1), t(0, 4, 1));
  EXPECT_EQ(projection(0, 1, 0), t(0, 5, 0));
  EXPECT_EQ(projection(1, 1, 0), t(1, 5, 0));
  EXPECT_EQ(projection(1, 3, 3), t(1, 7, 3));
}

TEST(FloatTensorTest, ProjectionAssign) {
  FloatTensor t = IotaTensor<8, 16>();
  uchen::core::AssignableProjection::dim_slice(t, 1, 8, 8) = IotaTensor<8, 8>();
  for (size_t i = t.elements() / 2; i < t.elements(); ++i) {
    EXPECT_EQ(t.get(i), i + 1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}