#include "uchen/rnn.h"

#include <array>
#include <cctype>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "uchen/layers.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/vector.h"

namespace uchen::testing {
namespace {

class TokenStream {
 public:
  using result_t = Vector<float, 'z' - 'a' + 4>;

  TokenStream(const char* input) : input_(input) {}
  TokenStream(std::string_view input) : input_(input) {}

  std::optional<result_t> Next() {
    size_t i = index_++;
    auto store = memory::ArrayStore<float, result_t::elements>::NewInstance();
    std::span<float, result_t::elements> span(&store->data()[0],
                                              result_t::elements);
    if (i == 0) {
      // Start token
      return result_t::OneHot(0, std::move(store));
    } else if (i == input_.size() + 1) {
      // End token
      return result_t::OneHot(1, std::move(store));
    } else if (i > input_.size() + 1) {
      // End of stream
      return std::nullopt;
    }
    char c = input_[i - 1];
    if (std::islower(c)) {
      return result_t::OneHot(c - 'a' + 3, std::move(store));
    }
    if (std::isupper(c)) {
      return result_t::OneHot(c - 'A' + 3, std::move(store));
    }
    return result_t::OneHot(2, std::move(store));
  }

 private:
  std::string_view input_;
  size_t index_ = 0;
};

TokenStream Emancipate(TokenStream&& stream) { return stream; }

TEST(RnnTest, TokenStream) {
  Model<InputLayer<TokenStream>> m;
  auto tokens = m("azAZ!", ModelParameters(&m, 0));
  std::vector<size_t> argmax;
  while (auto token = tokens.Next()) {
    EXPECT_THAT(*token, ::testing::SizeIs(29));
    argmax.push_back(token->ArgMax());
  }
  EXPECT_THAT(argmax, ::testing::ElementsAre(0, 3, 28, 3, 28, 2, 1));
}

struct TestStream {
  using value_type = Vector<float, 1>;

  class Iterator {
   public:
    explicit Iterator(size_t index) : index_(index) {}

    bool operator==(const Iterator& it) const { return index_ == it.index_; }

    Iterator& operator++() {
      ++index_;
      return *this;
    }

    value_type operator*() const { return {2}; }

   private:
    size_t index_ = 0;
  };

  Iterator begin() const { return Iterator(0); }
  Iterator end() const { return Iterator(2); }

  std::optional<Vector<float, 1>> Next() {
    if (ind_++ < 2) {
      return Vector<float, 1>({2});
    } else {
      return std::nullopt;
    }
  }

  int ind_ = 0;
};

TEST(RnnTest, Model) {
  constexpr Model m = layers::Rnn<TestStream, 2>(layers::Linear<2>);
  EXPECT_EQ(m.all_parameters_count(), 14);
  auto out =
      m({},
        {&m, {.1, -.1, .2, -.2, .3, -.3, .4, -.4, .5, -.5, .6, -.6, .7, -.7}});
  EXPECT_THAT(out, ::testing::ElementsAre(::testing::FloatEq(.635),
                                          ::testing::FloatEq(-.635)));
}

TEST(RnnTest, SpanInput) {
  constexpr Model m =
      layers::Rnn<std::span<Vector<float, 2>>, 1>(layers::Linear<1>);
  std::array<Vector<float, 2>, 2> input = {{{1, -1}, {2, -2}}};
  EXPECT_THAT(input[0], ::testing::ElementsAre(1, -1));
  Vector output = m(input, {&m, {0, .1, -.2, 2, 0, 1}});
  EXPECT_THAT(output, ::testing::ElementsAre(1.2));
}

}  // namespace
}  // namespace uchen::testing

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}