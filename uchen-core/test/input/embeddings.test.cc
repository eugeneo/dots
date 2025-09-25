#include <array>
#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "uchen/input/text.h"
#include "uchen/vector.h"

namespace uchen::input {
namespace {

class EmbeddingsMap {
 public:
  EmbeddingsMap() : embeddings_({{"a", {1, 1}}, {"c", {2, 2}}}) {}

  Vector<float, 2> Embed(std::string_view token) {
    auto it = embeddings_.find(token);
    if (it != embeddings_.end()) {
      return it->second;
    } else {
      return empty();
    }
  }

  Vector<float, 2> end() {
    Vector<float, 2> result = {7, 7};
    return result;
  }

  Vector<float, 2> empty() {
    Vector<float, 2> result = {9, 9};
    return result;
  }

 private:
  std::map<std::string, Vector<float, 2>, std::less<>> embeddings_;
};

using namespace std::string_view_literals;

TEST(EmbeddingTest, Embeds) {
  Encoder<4, EmbeddingsMap> embeds;
  EXPECT_THAT(embeds(std::vector{"a"sv, "b"sv, "c"sv}),
              ::testing::ElementsAre(1, 1, 9, 9, 2, 2, 9, 9));
}

TEST(EmbeddingTest, EmbedsEmpty) {
  Encoder<4, EmbeddingsMap> embeds;
  EXPECT_THAT(embeds(std::vector<std::string_view>{}),
              ::testing::ElementsAre(7, 7, 9, 9, 9, 9, 9, 9));
}

TEST(EmbeddingTest, EmbedsShort) {
  Encoder<3, EmbeddingsMap> embeds;
  EXPECT_THAT(embeds(std::vector{"a"sv}),
              ::testing::ElementsAre(1, 1, 7, 7, 9, 9));
}

TEST(EmbeddingTest, EmbedsTooLong) {
  Encoder<2, EmbeddingsMap> embeds;
  EXPECT_THAT(embeds(std::vector{"a"sv, "a"sv, "a"sv, "a"sv, "a"sv, "a"sv}),
              ::testing::ElementsAre(1, 1, 1, 1));
}

template <size_t C>
_Float16 CharacterValue(Vector<_Float16, C> v, size_t i) {
  std::valarray<_Float16> c(_Float16(0), 128);
  for (int i = 0; i < c.size(); ++i) {
    c[i] = i + 1;
  }
  const size_t classes = SourceFileTokenEncoder<_Float16, 1>::Classes;
  return (v.array()[std::slice(i * classes, classes, 1)] * c).sum();
}

template <size_t C>
std::vector<_Float16> EmbeddingValues(std::string_view token) {
  auto embedding = SourceFileTokenEncoder<_Float16, C>().Embed(token);
  std::vector<_Float16> expected;
  for (size_t i = 0; i < C; ++i) {
    expected.emplace_back(CharacterValue(embedding, i));
  }
  return expected;
}

TEST(SourceFileCharacterEmbedder, LowerCaseLetters) {
  EXPECT_THAT(EmbeddingValues<30>("abcdefghijklmnopqrstuvwxyz"),
              ::testing::ElementsAre(37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                     59, 60, 61, 62, 1, 0, 0, 0));
}

TEST(SourceFileCharacterEmbedder, UpperCaseLetters) {
  EXPECT_THAT(EmbeddingValues<30>("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
              ::testing::ElementsAre(37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                     59, 60, 61, 62, 1, 0, 0, 0));
}

TEST(SourceFileCharacterEmbedder, Digits) {
  EXPECT_THAT(EmbeddingValues<12>("0123456789"),
              ::testing::ElementsAre(4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 0));
}

TEST(SourceFileCharacterEmbedder, NonPrintableAndWhitespaces) {
  EXPECT_THAT(EmbeddingValues<34>("\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a"
                                  "\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14"
                                  "\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e"
                                  "\x1f\x20"),
              ::testing::ElementsAre(2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     2, 2, 2, 3, 1, 0));
}

TEST(SourceFileCharacterEmbedder, Rest) {
  std::string token;
  for (char c = ' ' + 1; c < 127; ++c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || std::isdigit(c)) {
      continue;
    }
    token.push_back(c);
  }
  EXPECT_THAT(
      EmbeddingValues<35>(token),
      ::testing::ElementsAre(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                             19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                             32, 33, 34, 35, 36, 1, 0, 0));
}

TEST(SourceFileCharacterEmbedder, TrimsToken) {
  EXPECT_THAT(EmbeddingValues<4>("     "), ::testing::ElementsAre(3, 3, 3, 1));
}

}  // namespace
}  // namespace uchen::input

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}