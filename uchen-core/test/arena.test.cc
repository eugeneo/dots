#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "gmock/gmock.h"
#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/vector.h"

namespace uchen {
namespace {

template <typename Model>
class IntermediatesPreservingModelContext {
 public:
  explicit IntermediatesPreservingModelContext(Model* /* deduction_only */) {}

  void* GetLayerMemory(int ind, size_t size, int align) { return nullptr; }

 private:
  template <size_t Starting = Model::kLayers - 1>
  static constexpr size_t BufferSize() {
    using ScratchSpace =
        typename Model::template Traits<Starting>::scratch_space_t;
    size_t s = 0;
    if constexpr (Starting != 0) {
      s = BufferSize<Starting - 1>();
    }
    return sizeof(ScratchSpace) + alignof(ScratchSpace) - 1 + s;
  }

  std::array<char, BufferSize()> buffer_;
};

TEST(ArenaTest, IntermediateResultsArena) {
  constexpr auto model = Model<Linear<Vector<float, 2>, 2>>() | layers::Relu |
                         layers::Linear<2> | layers::Relu | layers::Linear<2>;
  EXPECT_THAT(
      model({1, 2}, ModelParameters(&model, {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                             0, 0, 1, 0, 0, 1})),
      ::testing::ElementsAre(1, 2));
}

}  // namespace
}  // namespace uchen

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}