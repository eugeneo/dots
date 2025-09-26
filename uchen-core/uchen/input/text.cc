#include "uchen/input/text.h"

#include "absl/log/log.h"

namespace uchen::input {

namespace {

using namespace std::string_view_literals;

}

//
// class SourceFileTokenizer
//

CharacterAction SourceFileTokenizer::operator()(char c) {
  TokenType next = GetCharacterTokenType(c);
  if (next == current_type_) {
    return CharacterAction::Continue;
  }
  CharacterAction action = current_type_ == TokenType::Whitespace
                               ? CharacterAction::DropToken
                               : CharacterAction::EndToken;
  current_type_ = next;
  return action;
}

SourceFileTokenizer::TokenType SourceFileTokenizer::NextTokenType(
    TokenType type, char c) const {
  TokenType next_type = GetCharacterTokenType(c);
  if (next_type == TokenType::None) {
    return type;
  }
  switch (type) {
    case TokenType::Identifier:
      if (next_type == TokenType::Number) {
        return TokenType::Identifier;
      }
      break;
    case TokenType::Number:
      if (".e"sv.find(c) != std::string_view::npos) {
        return TokenType::Number;
      }
      break;
    case TokenType::Operator:
    case TokenType::Punct:
    case TokenType::Whitespace:
    case TokenType::None:
    case TokenType::Unknown:
      break;
  }
  return next_type;
}

SourceFileTokenizer::TokenType SourceFileTokenizer::GetCharacterTokenType(
    char c) const {
  if (std::isalpha(c) || "_"sv.find(c) != std::string_view::npos) {
    return TokenType::Identifier;
  }
  if (std::isdigit(c)) {
    return TokenType::Number;
  }
  if (std::isspace(c)) {
    return TokenType::Whitespace;
  }
  if ("!+-*/%&|^~<>=&"sv.find(c) != std::string_view::npos) {
    return TokenType::Operator;
  }
  if (".,;:?()[]{}#\"$'@\\"sv.find(c) != std::string_view::npos) {
    return TokenType::Punct;
  }
  LOG(FATAL) << "Unknown character: " << c;
  return TokenType::Unknown;
}

//
// class Tokenizer
//

std::vector<std::string_view> BaseTokenizer::operator()(
    std::string_view data) const {
  std::vector<std::string_view> tokens;
  size_t token_start = 0;
  size_t token_length = 0;
  auto processor = CreateProcessor();
  for (char c : data) {
    switch ((*processor)(c)) {
      case CharacterAction::Continue:
        token_length++;
        break;
      case CharacterAction::EndToken:
        if (token_length > 0) {
          tokens.emplace_back(data.substr(token_start, token_length));
        }
        // Fallthrough.
      case CharacterAction::DropToken:
        token_start += token_length;
        token_length = 1;
        break;
    }
  }
  if (token_length > 0) {
    tokens.emplace_back(data.substr(token_start, token_length));
  }
  return tokens;
}

}  // namespace uchen::input