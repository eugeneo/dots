#include "uchen/tensor/special_tensors.h"

#include <array>
#include <cstddef>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"  // IWYU pragma: keep

#include "test/tensor/tensor_test_util.h"
#include "uchen/tensor/float_tensor.h"
#include "uchen/tensor/function.h"

namespace {

using uchen::core::FloatTensor;
using uchen::core::testing::IotaTensor;

TEST(RowProjectionsTensor, Basic) {
  std::array<float, 8> data;
  std::iota(data.begin(), data.end(), 5.0f);
  uchen::core::RowProjectionsTensor<2, 4, 8> tensor;
  for (auto& span : tensor.flat_data()) {
    span = data;
  }
  for (size_t i = 0; i < 8; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      EXPECT_EQ(tensor.get(i * 8 + j), data[j]) << "i: " << i << ", j: " << j;
    }
  }
}

TEST(TransposeTest, Transpose) {
  FloatTensor tensor = IotaTensor<2, 4, 8>();
  uchen::core::details::TransposeTranslator translator;
  auto transposed = uchen::core::Transpose(tensor);
  EXPECT_EQ(transposed.rank(), 3);
  EXPECT_EQ(transposed.dim(0), 2);
  EXPECT_EQ(transposed.dim(1), 8);
  EXPECT_EQ(transposed.dim(2), 4);
  EXPECT_EQ(translator.Translate(0, tensor), 0);
  EXPECT_EQ(transposed(0, 0, 0), 1);
  EXPECT_EQ(translator.Translate(4, tensor), 1);
  EXPECT_EQ(transposed(0, 1, 0), tensor(0, 0, 1));
  EXPECT_EQ(transposed(0, 2, 0), tensor(0, 0, 2));
  EXPECT_EQ(transposed(0, 0, 1), tensor(0, 1, 0));
  EXPECT_EQ(transposed(1, 0, 1), tensor(1, 1, 0));
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}