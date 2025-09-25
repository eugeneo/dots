#ifndef EXPERIMENTAL_RNN_NAMES_H
#define EXPERIMENTAL_RNN_NAMES_H

#include <filesystem>
#include <memory>
#include <string>
#include <valarray>
#include <vector>

#include "uchen/layers.h"
#include "uchen/model.h"
#include "uchen/rnn.h"
#include "uchen/softmax.h"

namespace uchen::experimental {

namespace internal {

class Input {
 public:
  using value_type = uchen::Vector<float, 'z' - 'a' + 4>;

  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = typename Input::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    explicit Iterator(std::string_view data, size_t ind)
        : data_(data), ind_(ind) {}

    Iterator& operator++() {
      ++ind_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const Iterator& other) const { return ind_ == other.ind_; }

    bool operator!=(const Iterator& other) const { return !(*this == other); }

    value_type operator*() const {
      auto store = memory::ArrayStore<value_type::value_type,
                                      value_type::elements>::NewInstance();
      if (ind_ == 0) {
        return value_type::OneHot(0, std::move(store));
      }
      if (ind_ == data_.length() + 1) {
        return value_type::OneHot(1, std::move(store));
      }
      char c = std::tolower(data_[ind_]);
      if (c >= 'a' && c <= 'z') {
        return value_type::OneHot(c - 'a' + 3, std::move(store));
      }
      return value_type::OneHot(2, std::move(store));
    }

   private:
    std::string_view data_;
    size_t ind_;
  };

  Input(std::string_view name) : name_(name) {}
  Iterator begin() const { return Iterator(name_, 0); }
  Iterator end() const { return Iterator(name_, name_.length()); }
  std::string_view str() const { return name_; }

 private:
  std::string_view name_;
};

template <typename T, T... V>
constexpr std::array<T, sizeof...(V)> MakeArray(
    std::integer_sequence<T, V...> /* seq */) {
  return std::array<T, sizeof...(V)>({(V + 'a')...});
}

}  // namespace internal

constexpr uchen::Model kNameRnn =
    uchen::layers::Rnn<internal::Input, 50>(
        uchen::layers::Linear<10> | uchen::layers::Relu |
        uchen::layers::Linear<10> | uchen::layers::Relu) |
    uchen::layers::Categories(
        internal::MakeArray(std::make_integer_sequence<char, 'z' - 'a' + 2>()));

std::vector<std::pair<const internal::Input, const char>>
PrepareTrainingDataSet(std::span<const std::string> names);

}  // namespace uchen::experimental

namespace std {

std::ostream& operator<<(std::ostream& os,
                         uchen::experimental::internal::Input /* input */);

}  // namespace std

#endif  // EXPERIMENTAL_RNN_NAMES_H