#include <algorithm>
#include <cctype>
#include <cstddef>
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/log.h"

#include "gmock/gmock.h"
#include "uchen/input/text.h"

namespace uchen::input::testing {

namespace {

constexpr std::array kKeywords = {
    "import",  "include",  "public",   "in",        "return",  "void",
    "or",      "is",       "for",      "this",      "of",      "and",
    "const",   "not",      "type",     "class",     "new",     "with",
    "std",  // Important for c++!
    "final",   "use",      "at",       "if",        "private", "static",
    "except",  "String",   "int",      "namespace", "string",  "endif",
    "package", "extern",   "delegate", "true",      "char",    "nullptr",
    "define",  "override", "null",     "auto",      "method",  "from",
    "ifndef",  "throws",   "move",     "bool",      "byte",    "ifdef",
    "error",   "extends",  "false",    "def",
};

constexpr std::array kWords = {
    "the",
    "License",
    "grpc",
    "java",
    "import",
    "include",
    "under",
    "h",
    "public",
    "io",
    "in",
    "return",
    "void",
    "to",
    "or",
    "language",
    "may",
    "is",
    "distributed",
    "a",
    "src",
    "for",
    "this",
    "org",
    "of",
    "core",
    "and",
    "const",
    "com",
    "not",
    "type",
    "class",
    "by",
    "file",
    "new",
    "api",
    "with",
    "path",
    "size",
    "an",
    "std",
    "cpp",
    "https",
    "github",
    "destination",
    "blob",
    "blobs",
    "final",
    "git",
    "mode",
    "on",
    "repos",
    "sha",
    "url",
    "use",
    "at",
    "gRPC",
    "if",
    "private",
    "static",
    "http",
    "Override",
    "required",
    "you",
    "either",
    "except",
    "See",
    "copy",
    "permissions",
    "writing",
    "ANY",
    "AS",
    "Apache",
    "BASIS",
    "CONDITIONS",
    "Copyright",
    "IS",
    "KIND",
    "LICENSE",
    "Licensed",
    "OF",
    "OR",
    "Unless",
    "Version",
    "WARRANTIES",
    "WITHOUT",
    "You",
    "agreed",
    "apache",
    "applicable",
    "compliance",
    "express",
    "governing",
    "implied",
    "law",
    "licenses",
    "limitations",
    "obtain",
    "software",
    "specific",
    "www",
    "lib",
    "String",
    "The",
    "int",
    "absl",
    "namespace",
    "be",
    "link",
    "internal",
    "xds",
    "assertThat",
    "s",
    "string",
    "endif",
    "grpc_core",
    "test",
    "Test",
    "package",
    "Authors",
    "extern",
    "util",
    "that",
    "i",
    "junit",
    "delegate",
    "name",
    "true",
    "upb",
    "google",
    "status",
    "support",
    "common",
    "char",
    "config",
    "authors",
    "main",
    "nullptr",
    "define",
    "override",
    "transport",
    "null",
    "buffer",
    "n",
    "v3",
    "auto",
    "code",
    "method",
    "This",
    "from",
    "will",
    "ext",
    "checkNotNull",
    "ifndef",
    "key",
    "Status",
    "throws",
    "move",
    "port",
    "bool",
    "value",
    "string_view",
    "args",
    "envoy",
    "p",
    "port_platform",
    "when",
    "add",
    "headers",
    "Metadata",
    "byte",
    "JUnit4",
    "Map",
    "RunWith",
    "ConnectivityState",
    "ifdef",
    "isEqualTo",
    "it",
    "C",
    "error",
    "EXPECT_EQ",
    "LoadBalancer",
    "inc",
    "extends",
    "mock",
    "should",
    "false",
    "x00",
    "Attributes",
    "T",
    "are",
    "base",
    "security",
    "service",
    "cc",
    "gprpp",
    "helper",
    "used",
    "__cplusplus",
    "call",
    "def",
};

using TokensHistogram = std::map<std::string, size_t, std::less<>>;

TokensHistogram GetFileTokensHistogram(
    const std::filesystem::directory_entry& entry) {
  if (entry.is_directory()) {
    return TokensHistogram();
  }
  std::ifstream file(entry.path());
  std::string data;
  TokensHistogram histogram;
  Tokenizer<SourceFileTokenizer> tokenizer;
  while (std::getline(file, data)) {
    for (const auto& token : tokenizer(data)) {
      if (token.empty() || !std::isalpha(token[0])) {
        continue;
      }
      auto it = histogram.find(token);
      if (it == histogram.end()) {
        histogram.emplace(token, 1);
      } else {
        it->second++;
      }
    }
  }
  return histogram;
}

struct SortByCount {
  bool operator()(const std::pair<std::string, size_t>& a,
                  const std::pair<std::string, size_t>& b) const {
    return a.second > b.second || (a.second == b.second && a.first < b.first);
  }
};

void PrintTokenCounts(const TokensHistogram& histogram,
                      const std::set<std::string_view>& ignored) {
  std::set<std::pair<std::string, size_t>, SortByCount> sorted;
  for (const auto& [token, count] : histogram) {
    if (ignored.find(token) == ignored.end()) {
      sorted.emplace(token, count);
    }
  }
  size_t i = 0;
  for (const auto& [token, count] : sorted) {
    std::cout << token << "\n";
    if (++i > 100) {
      LOG(INFO) << "Least common token occurrences: " << count;
      break;
    }
  }
}

namespace testing {

TEST(TokenizerTest, DISABLED_CppTokens) {
  std::filesystem::recursive_directory_iterator it(
      "/home/eostroukhov/code/uchen/lang-training");
  std::vector<TokensHistogram> histograms;
  std::transform(std::filesystem::begin(it), std::filesystem::end(it),
                 std::back_inserter(histograms), GetFileTokensHistogram);
  auto all_counts =
      std::reduce(histograms.begin(), histograms.end(), TokensHistogram(),
                  [](TokensHistogram a, const TokensHistogram& b) {
                    for (const auto& [token, count] : b) {
                      a[token] += count;
                    }
                    return a;
                  });

  PrintTokenCounts(all_counts,
                   std::set<std::string_view>(kWords.begin(), kWords.end()));
  std::set<char> all_chars;
  for (std::string_view word : kKeywords) {
    for (char c : word) {
      all_chars.insert(c);
    }
  }
  for (char c = 'a'; c <= 'z'; c++) {
    if (all_chars.find(c) == all_chars.end()) {
      LOG(INFO) << "Unused " << c << std::endl;
    }
  }
  for (char c = 'A'; c <= 'Z'; c++) {
    if (all_chars.find(c) == all_chars.end()) {
      LOG(INFO) << "Unused " << c << std::endl;
    }
  }
  EXPECT_FALSE(true);
}

}  // namespace testing

TEST(TokenizerTest, TokenizeLine1) {
  Tokenizer<SourceFileTokenizer> tokenizer;
  auto result = tokenizer("int main() { return 0; }");
  EXPECT_THAT(result, ::testing::ElementsAre("int", "main", "()", "{", "return",
                                             "0", ";", "}"));
}

TEST(TokenizerTest, TokenizeInclude) {
  Tokenizer<SourceFileTokenizer> tokenizer;
  EXPECT_THAT(tokenizer("#include"), ::testing::ElementsAre("#", "include"));
}

TEST(TokenizerTest, DISABLED_Cyrillic) {}
TEST(TokenizerTest, DISABLED_Chinese) {}

}  // namespace

}  // namespace uchen::input::testing