#include "uchen/training/training.h"

#include <array>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/strings/str_format.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/parameters.h"
#include "uchen/vector.h"

namespace uchen::training::testing {

namespace {

constexpr std::array<std::pair<int, int>, 10> arr = {{
    {11, 21},
    {12, 22},
    {13, 23},
    {14, 24},
    {15, 25},
    {16, 26},
    {17, 27},
    {18, 28},
    {19, 29},
    {10, 20},
}};

TEST(TrainingTest, Grad) {
  auto model =
      layers::Input<Vector<float, 1>> | layers::Linear<1>;  // y = 2x + 1
  TrainingData<Vector<float, 1>, Vector<float, 1>> arr = {
      {{0}, {1}},
      {{1}, {3}},
      {{2}, {5}},
  };
  Training training(&model, ModelParameters(&model, {-2.f, 2}));
  EXPECT_EQ(training.Loss(arr), 9);
  training = training.Generation(arr, 0.1);
  EXPECT_THAT(training.parameters(), ::testing::ElementsAre(-1.4, 2.6));
  EXPECT_NEAR(training.Loss(arr), 3.48, 0.001);
}

class MoveOnly {
 public:
  MoveOnly() = default;
  MoveOnly(const MoveOnly& /* other */) = delete;
  MoveOnly(MoveOnly&& /* other */) = default;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const MoveOnly& p) {
    absl::Format(&sink, "MoveOnly");
  }
};

TEST(TrainingDataTest, MoveOnly) {
  TrainingData data = TrainingData<MoveOnly, int>::FromArray<2>(
      {{{MoveOnly(), 1}, {MoveOnly(), 2}}});
  EXPECT_THAT(data, ::testing::SizeIs(2));
}

TEST(TrainingDataTest, Split) {
  TrainingData<int, int> data(arr);
  auto [training, validation] = data.Split(.8);
  EXPECT_THAT(training, ::testing::ElementsAre(arr[0], arr[1], arr[2], arr[3],
                                               arr[4], arr[5], arr[6], arr[7]));
  EXPECT_THAT(validation, ::testing::ElementsAre(arr[8], arr[9]));
}

TEST(TrainingDataTest, Shuffle) {
  TrainingData data = TrainingData<int, int>(arr).Shuffle();
  EXPECT_THAT(data, ::testing::UnorderedElementsAre(
                        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6],
                        arr[7], arr[8], arr[9]));
}

TEST(TrainingDataTest, Batches) {
  std::vector batches = TrainingData<int, int>(arr).BatchWithSize(4);
  EXPECT_THAT(batches,
              ::testing::ElementsAre(
                  ::testing::ElementsAre(arr[0], arr[1], arr[2], arr[3]),
                  ::testing::ElementsAre(arr[4], arr[5], arr[6], arr[7]),
                  ::testing::ElementsAre(arr[8], arr[9])));
  batches = TrainingData<int, int>(arr).BatchWithSize(5);
  EXPECT_THAT(
      batches,
      ::testing::ElementsAre(
          ::testing::ElementsAre(arr[0], arr[1], arr[2], arr[3], arr[4]),
          ::testing::ElementsAre(arr[5], arr[6], arr[7], arr[8], arr[9])));
}

}  // namespace

}  // namespace uchen::training::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}