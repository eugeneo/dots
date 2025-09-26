#include <array>
#include <cctype>
#include <cstddef>
#include <memory>
#include <string_view>
#include <tuple>
#include <valarray>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "gmock/gmock.h"
#include "test/test_lib.h"
#include "uchen/input/text.h"
#include "uchen/layers.h"
#include "uchen/model.h"

namespace uchen::input {

namespace {

TEST(TextE2ETest, TokenizerLayer) {
  auto model1 = layers::TextTokenizer<SourceFileTokenizer>;
  ModelParameters parameters(&model1, {});
  EXPECT_THAT(model1("int main() { return 0; }", parameters),
              ::testing::ElementsAre("int", "main", "()", "{", "return", "0",
                                     ";", "}"));
  auto model2 =
      model1 | layers::TokenEncoder<10, SourceFileTokenEncoder<float, 5>>;
  ModelParameters parameters2(&model2, {});
  auto encoded = model2("int main() { return 0; }", parameters2);
  EXPECT_EQ(encoded.size(), 3200);
  std::set<size_t> non_zeroes = {44,   113,  183,  192,  368,  420,  492,  561,
                                 576,  651,  716,  768,  992,  1024, 1333, 1384,
                                 1463, 1528, 1536, 1603, 1664, 1940, 1984, 2274,
                                 2304, 2561, 2880, 2944, 3008, 3072, 3136};
  for (size_t i = 0; i < encoded.size(); ++i) {
    EXPECT_EQ(encoded[i], non_zeroes.find(i) == non_zeroes.end() ? 0 : 1)
        << "i = " << i;
  }
}

TEST(TextE2ETest, TokenizerToOnehot) {
  auto model = layers::TextTokenizer<SourceFileTokenizer> |
               layers::TokenEncoder<10, SourceFileTokenEncoder<float, 5>> |
               ::uchen::layers::Linear<2>;
  EXPECT_EQ(model.all_parameters_count(), 6402);
  std::vector<std::vector<float>> data;
  data.resize(2);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i].resize(3201);
    for (size_t j = 0; j < data[i].size(); ++j) {
      data[i][j] = static_cast<float>(data[i].size() * i + j) /
                   model.all_parameters_count();
    }
  }
  ModelParameters model_parameters(&model, testing::RearrangeLinear(data));
  auto encoded = model("int main() { return 0; }", model_parameters);
  EXPECT_FLOAT_EQ(encoded[0], 7.3280225);
  EXPECT_FLOAT_EQ(encoded[1], 23.328026);
}

}  // namespace
}  // namespace uchen::input

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}