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
#include "uchen/tensor/special_tensors.h"

namespace {

using uchen::core::FloatTensor;
using uchen::core::testing::IotaTensor;

TEST(FloatTensor, ByIdentity) {
  FloatTensor a = IotaTensor<4, 4>();
  FloatTensor b =
      uchen::core::testing::Generate<4, 4>([](std::span<const size_t> index) {
        return index.back() == index[index.size() - 2] ? 1.0f : 0.0f;
      });
  FloatTensor<4, 4> c = a * b;
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}