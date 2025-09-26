#ifndef UCHEN_INPUT_TEXT_H_
#define UCHEN_INPUT_TEXT_H_

#include <cctype>
#include <map>
#include <memory>
#include <set>
#include <string_view>
#include <vector>

#include "absl/log/log.h"

#include "uchen/layers.h"

namespace uchen::input {

enum class CharacterAction { Continue, EndToken, DropToken };

class ByCharacterTokenizer {
 public:
  virtual ~ByCharacterTokenizer() = default;
  virtual CharacterAction operator()(char c) = 0;
};

class SourceFileTokenizer : public ByCharacterTokenizer {
 public:
  CharacterAction operator()(char c);

 private:
  enum class TokenType {
    Identifier,
    Number,
    Operator,
    Punct,
    Whitespace,
    None,
    Unknown
  };

  TokenType NextTokenType(TokenType type, char c) const;
  TokenType GetCharacterTokenType(char c) const;
  TokenType current_type_ = TokenType::None;
};

class BaseTokenizer {
 public:
  using input_t = std::string_view;
  using output_t = std::vector<std::string_view>;
  constexpr static size_t parameter_count = 0;

  std::vector<std::string_view> operator()(std::string_view data) const;

  virtual std::unique_ptr<ByCharacterTokenizer> CreateProcessor() const = 0;
};

template <typename CharacterProcessor>
class Tokenizer : public BaseTokenizer {
 public:
  std::unique_ptr<ByCharacterTokenizer> CreateProcessor() const override {
    return std::make_unique<CharacterProcessor>();
  }
};

template <size_t InputLen, typename TokenEmbedder>
  requires(InputLen > 0)
class Encoder {
 public:
  using ProviderReturn = decltype(std::declval<TokenEmbedder>().Embed(""));
  constexpr static size_t EmbeddingSize = ProviderReturn::elements;
  using input_t = std::vector<std::string_view>;
  using output_t = typename ProviderReturn::template repeated_t<InputLen>;
  constexpr static size_t parameter_count = 0;

  output_t operator()(const input_t& tokens) const {
    output_t result;
    TokenEmbedder embedder;
    size_t i;
    for (i = 0; i < InputLen && i < tokens.size(); ++i) {
      result.Assign(i * EmbeddingSize, embedder.Embed(tokens[i]));
    }
    if (i < InputLen - 1) {
      result.Assign(i++ * EmbeddingSize, embedder.end());
    }
    for (; i < InputLen; ++i) {
      result.Assign(i * EmbeddingSize, embedder.empty());
    }
    return result;
  }
};

template <typename V, size_t TokenLen>
class SourceFileTokenEncoder {
 public:
  static constexpr size_t Classes = 64;
  using Embedding = Vector<V, Classes * TokenLen>;

  Embedding Embed(std::string_view token) {
    Embedding result;
    size_t i = 0;
    for (; i < std::min(TokenLen - 1, token.size()); ++i) {
      result[i * Classes + CharacterCode(token[i])] = 1;
    }
    result[std::min(TokenLen - 1, token.size()) * Classes] = 1;
    return result;
  }

  Embedding empty() {
    Embedding result;
    for (size_t i = 0; i < TokenLen; ++i) {
      result[i * Classes] = 1;
    }
    return result;
  }

  Embedding end() {
    Embedding result;
    result[1] = 1;
    return result;
  }

 private:
  static size_t CharacterCode(int c) {
    // 0: eof, 1: unknown, 2: whitespace, 3: digit, 4 - 36: symbol,
    // 37 - 63: letter
    if (std::isblank(c)) {
      return 2;
    } else if (c < 0x20) {
      return 1;
    } else if (c < '0') {
      return c - 29;
    } else if (c <= '9') {
      return 3;
    } else if (c < 'A') {
      return c - 39;
    } else if (c <= 'Z') {
      return c - 29;
    } else if (c < 'a') {
      return c - 65;
    } else if (c <= 'z') {
      return c - 61;
    } else if (c < 127) {
      return c - 91;
    } else {
      return 1;
    }
  }
};

namespace layers {

template <size_t Tokens, typename EmbeddingsProvider>
class TokenEncoderDesc {
 public:
  constexpr TokenEncoderDesc() = default;

  template <typename Layer>
  constexpr Encoder<Tokens, EmbeddingsProvider> stack(
      const Layer& /* layer */) const {
    return {};
  }
};

template <size_t Tokens, typename PerTokenEncoder>
constexpr Layer<TokenEncoderDesc<Tokens, PerTokenEncoder>> TokenEncoder;

template <typename CharacterProcessor>
constexpr Model<Tokenizer<CharacterProcessor>> TextTokenizer;

}  // namespace layers

}  // namespace uchen::input

#endif  // UCHEN_INPUT_TEXT_H_