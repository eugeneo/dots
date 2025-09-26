#include "uchen/text/embeddings.h"

#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "uchen/parameters.h"
#include "uchen/text/embeddings_training.h"  // IWYU pragma: keep
#include "uchen/training/model_gradients.h"

namespace uchen::text::testing {

TEST(EmbeddingsTest, Test) {
  Model model = Embeddings<3, 3, 3>;
  EXPECT_EQ(model.all_parameters_count(), 9);
  std::array<uint32_t, 3> tokens = {0, 2, 0};
  ModelParameters parameters(&model, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  EXPECT_THAT(model(tokens, parameters),
              ::testing::ElementsAre(1, 2, 3, 7, 8, 9, 1, 2, 3));
}

TEST(EmbeddingsGradientsTest, TEST) {
  Model model = Embeddings<3, 3, 3>;
  typename uchen::training::ForwardPassResult<decltype(model),
                                              std::span<const uint32_t, 3>>::
      ContextForGradientCalculation context(&model);
  std::array<uint32_t, 3> tokens = {0, 2, 0};
  std::span<const uint32_t, 3> tokens_span(tokens);
  ModelParameters p(&model, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto r = model(tokens_span, p, context);
  training::ForwardPassResult fpr(&model, tokens_span, p);
  EXPECT_THAT(
      fpr.CalculateParameterGradients({1, 2, 3, 4, 5, 6, 7, 8, 9}).second,
      ::testing::ElementsAre(8, 10, 12, 0, 0, 0, 4, 5, 6));
}

}  // namespace uchen::text::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}