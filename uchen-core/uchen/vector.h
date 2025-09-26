#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include "absl/strings/str_format.h"

#include "uchen/layer_traits.h"
#include "uchen/math/matrix.h"
#include "uchen/memory.h"

#ifndef UCHEN_VECTOR_H
#define UCHEN_VECTOR_H

namespace uchen {

template <typename V, size_t C>
  requires(C > 0)  // Might need to be dropped. I just like this check.
class Vector {
 public:
  using value_type = V;
  template <size_t T>
  using repeated_t = Vector<V, C * T>;
  template <size_t T>
    requires(C % T == 0)
  using split_t = Vector<V, C / T>;
  template <size_t S>
  using with_size = Vector<V, S>;
  template <int S>
  using resize = Vector<V, C + S>;

  constexpr static size_t elements = C;

  Vector() : Vector(V{0}) {}

  explicit Vector(const V& v)
      : Vector(memory::ArrayStore<V, C>::NewInstance(v)) {}

  Vector(std::initializer_list<V> init)
      : Vector(memory::ArrayStore<V, C>::NewInstance(init)) {}

  explicit Vector(std::unique_ptr<memory::ArrayStore<V, C>> store)
      : data_(store->data()) {
    store_ = std::move(store);
  }

  Vector(std::span<const V, C> data,
         std::unique_ptr<memory::Deletable> store = nullptr)
      : data_(data), store_(std::move(store)) {}

  template <typename S>
  Vector(const math::Matrix<S>& matrix) : Vector(matrix.data_view(), nullptr) {}

  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
  constexpr size_t size() const { return C; }

  const V& operator[](size_t i) const { return data_[i]; }

  std::span<const V, C> data() const { return data_; }

  template <typename V1, size_t C1>
  bool operator==(const Vector<V1, C1>& other) const {
    if (C != C1) {
      return false;
    }
    for (size_t i = 0; i < C; ++i) {
      if ((*this)[i] != other.values_[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const V& v) const { return C == 1 && (*this)[0] == v; }

  V sum() const {
    V result = V{0};
    for (const V& v : data_) {
      result += v;
    }
    return result;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Vector& category) {
    absl::Format(&sink, "V{");
    for (size_t i = 0; i < C; ++i) {
      absl::Format(&sink, "%v", static_cast<float>(category[i]));
      if (i < C - 1) {
        absl::Format(&sink, ", ");
      }
    }
    absl::Format(&sink, "}");
  }

  size_t ArgMax() const {
    size_t result = 0;
    for (size_t i = 1; i < C; ++i) {
      if (data_[i] > data_[result]) {
        result = i;
      }
    }
    return result;
  }

  template <size_t C1>
  Vector<V, C + C1> Join(
      const Vector<V, C1>& other,
      std::unique_ptr<memory::ArrayStore<V, C + C1>> s) const {
    std::span data = s->data();
    return Join(other, data, std::move(s));
  }

  template <size_t C1>
  Vector<V, C + C1> Join(
      const Vector<V, C1>& other, std::span<V, C + C1> data,
      std::unique_ptr<memory::Deletable> store = nullptr) const {
    std::copy(begin(), end(), data.begin());
    std::copy(other.begin(), other.end(), data.begin() + C);
    return Vector<V, C + C1>(data, std::move(store));
  }

  Vector Emancipate() const {
    return Vector(std::make_unique<memory::ArrayStore<V, C>>(data_));
  }

  static Vector OneHot(size_t index,
                       std::unique_ptr<memory::ArrayStore<V, C>> store) {
    std::span data = store->data();
    return OneHot(index, data, std::move(store));
  }

  static Vector OneHot(size_t index, std::span<V, C> data,
                       std::unique_ptr<memory::Deletable> store) {
    for (size_t i = 0; i < C; ++i) {
      data[i] = index == i ? 1 : 0;
    }
    return Vector(data, std::move(store));
  }

  operator std::span<const V, C>() const { return data_; }

  template <uint32_t R, uint32_t C1>
    requires(C == static_cast<size_t>(R) * C1)
  operator math::RowMajorView<R, C1>() const {
    return math::RowMajorView<R, C1>(data_);
  }

 private:
  std::span<const V, C> data_;
  std::shared_ptr<memory::Deletable> store_;
};

template <typename V, size_t C>
static Vector<V, C> Emancipate(const Vector<V, C>& result) {
  return result.Emancipate();
}

template <size_t I, typename V, size_t C>
  requires(I < C)
constexpr const V get(const uchen::Vector<V, C>& vector) noexcept {
  return vector[I];
}

template <typename V, size_t PC = 0>
struct VectorOutputLayer
    : public LayerTraitFields<
          V, PC, memory::ArrayStore<typename V::value_type, V::elements>> {};

}  // namespace uchen

namespace std {

template <typename V, size_t C>
struct tuple_size<::uchen::Vector<V, C>>
    : public std::integral_constant<size_t, C> {};

template <size_t I, typename V, size_t C>
  requires(I < C)
struct tuple_element<I, uchen::Vector<V, C>> {
  using type = V;
};

template <typename V, size_t C>
std::ostream& operator<<(std::ostream& os, const uchen::Vector<V, C>& vector) {
  os << absl::StrFormat("%v", vector);
  return os;
}

}  // namespace std

#endif  // UCHEN_VECTOR_H
