#include "uchen/training/kaiming_he.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/strings/str_cat.h"

#include "gmock/gmock.h"
#include "uchen/linear.h"
#include "uchen/parameters.h"
#include "uchen/rnn.h"

namespace uchen::training::testing {

namespace {

MATCHER_P2(IsBetween, a, b,
           absl::StrCat(negation ? "isn't" : "is", " between ", a, " and ",
                        b)) {
  return a <= arg && arg <= b;
}

TEST(KaimingHeTest, RnnAndLinearNotRandom) {
  Model m = layers::Rnn<std::span<Vector<float, 2>>, 1>(layers::Linear<2>) |
            layers::Linear<2>;
  ModelParameters parameters =
      KaimingHeInitializedParameters(&m, []() { return 1; });
  EXPECT_THAT(parameters.layer_parameters<0>(),
              ::testing::ElementsAre(0, 0, .8, .8, .8, .8, .8, .8, 0,
                                     ::testing::FloatNear(1.33, 0.01),
                                     ::testing::FloatNear(1.33, 0.01)));
  EXPECT_THAT(parameters.layer_parameters<1>(),
              ::testing::ElementsAre(0, 0, 1, 1, 1, 1));
}

TEST(KaimingHeTest, RandomParameters) {
  Model<Linear<Vector<float, 100>, 100>> m;
  ModelParameters p = KaimingHeInitializedParameters(&m);
  float prev = *p.begin();
  size_t diff_count = 0;
  for (auto i = p.begin(); i < p.end(); ++i) {
    if (prev != *i) {
      diff_count++;
    }
  }
  EXPECT_GT(diff_count, 80) << p;
}

}  // namespace

}  // namespace uchen::training::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}