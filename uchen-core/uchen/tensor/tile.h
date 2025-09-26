#ifndef UCHEN_TENSOR_TILE_H
#define UCHEN_TENSOR_TILE_H

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ostream>
#include <span>
#include <type_traits>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"  // IWYU pragma: keep
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "hwy/highway.h"
#include "hwy/print-inl.h"

namespace uchen::tensor {

template <typename T>
class Tile4x4;

namespace hn = ::hwy::HWY_NAMESPACE;

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {

template <typename T>
static auto LoadImpl(const auto& tile, const auto& delegate) {
  hn::FixedTag<T, 4> d;
  return tile.LoadHighway(d, delegate);
}

}  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

namespace ops {

template <typename T>
class TileOp {
 public:
  virtual ~TileOp() = default;
  virtual void Apply(Tile4x4<T>& tile) const = 0;
};

void CheckNoOverlap(const auto& a, const auto& b) {
  CHECK(a.data() + a.size() <= b.data() || b.data() + b.size() <= a.data());
}

template <typename A, typename B>
class AddOp {
 public:
  AddOp(const A& a, const B& b) : a_(a), b_(b) {}

  template <typename D>
  void LoadHighway(const D& d, const auto& delegate) const {
    return a_.LoadHighway(
        d, [&]<typename... V1s>(const D& d1, const V1s&... v1) {
          return b_.LoadHighway(
              d, [&]<typename... V2s>(const D& d2, const V2s&... v2) {
                return delegate(d1, hn::Add(v1, v2)...);
              });
        });
  }

 private:
  A a_;
  B b_;
};

template <typename T>
class ScalarMulOp {
 public:
  explicit ScalarMulOp(T a, typename T::value_type b)
      : a_(std::move(a)), b_(b) {}

  template <typename D>
  void LoadHighway(const D& d, const auto& delegate) const {
    return a_.LoadHighway(d, [&]<typename... Vs>(const D& d1, const Vs&... vs) {
      auto m = hn::Set(d1, b_);
      return delegate(d1, hn::Mul(vs, m)...);
    });
  }

 private:
  T a_;
  typename T::value_type b_;
};

template <typename T>
class MulOp : public TileOp<T> {
 public:
  explicit MulOp(Tile4x4<T> a, Tile4x4<T> b)
      : a_(std::move(a)), b_(std::move(b)) {}

  void Apply(Tile4x4<T>& tile) const override {
    CheckNoOverlap(a_.data(), tile.data());
    CheckNoOverlap(b_.data(), tile.data());
    std::span data = tile.data();
    std::fill(data.begin(), data.end(), 0);
    for (size_t r = 0; r < Tile4x4<T>::rows; ++r) {
      for (size_t c = 0; c < Tile4x4<T>::columns; ++c) {
        for (size_t i = 0; i < Tile4x4<T>::rows; ++i) {
          tile[r, c] += a_[r, i] * b_[i, c];
        }
      }
    }
  }

 private:
  Tile4x4<T> a_;
  Tile4x4<T> b_;
};

template <typename Tile>
class TransposeOp {
 public:
  explicit TransposeOp(const Tile& tile) : tile_(tile) {}

  void LoadHighway(const auto& d, const auto& delegate) const {
    tile_.LoadHighway(
        d, [&]<typename D>(D d1, hn::VFromD<D> row0, hn::VFromD<D> row1,
                           hn::VFromD<D> row2, hn::VFromD<D> row3) {
          using V = hn::VFromD<D>;
          V t0 = hn::InterleaveLower(d, row0, row1);
          V t1 = hn::InterleaveLower(d, row2, row3);
          V c0 = hn::ConcatLowerLower(d, t1, t0);
          V c1 = hn::ConcatUpperUpper(d, t1, t0);
          t0 = hn::InterleaveUpper(d, row0, row1);
          t1 = hn::InterleaveUpper(d, row2, row3);
          V c2 = hn::ConcatLowerLower(d, t1, t0);
          V c3 = hn::ConcatUpperUpper(d, t1, t0);
          return delegate(d, c0, c1, c2, c3);
        });
  }

 private:
  Tile tile_;
};

}  // namespace ops

template <typename T>
class Tile4x4 {
 public:
  using value_type = T;
  constexpr static size_t elements = 16;
  constexpr static size_t rows = 4;
  constexpr static size_t columns = 4;

  constexpr Tile4x4(std::span<T> data) : data_(data.template first<16>()) {}

  Tile4x4& operator=(const Tile4x4& other) {
    Assign(other);
    return *this;
  }

  template <typename Tile>
    requires std::negation_v<std::is_convertible<Tile, T>> &&
             std::negation_v<std::is_base_of<ops::TileOp<T>, Tile>>
  Tile4x4& operator=(const Tile& other) {
    Assign(other);
    return *this;
  }

  Tile4x4& operator=(T t) {
    std::fill(data_.begin(), data_.end(), t);
    return *this;
  }

  Tile4x4& operator=(const ops::TileOp<T>& op) {
    op.Apply(*this);
    return *this;
  }

  Tile4x4& operator*=(T t) { return operator=(ops::ScalarMulOp(*this, t)); }

  Tile4x4& operator+=(const Tile4x4& other) {
    return operator=(ops::AddOp(*this, other));
  }

  bool operator==(const std::array<std::array<T, 4>, 4>& other) const {
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < columns; ++c) {
        if (data_[r * columns + c] != other[r][c]) {
          return false;
        }
      }
    }
    return true;
  }

  auto begin() const { return data_.begin(); }

  auto end() const { return data_.end(); }

  std::span<T, elements> data() { return data_; }

  std::span<const T, elements> data() const { return data_; }

  T& operator[](size_t row, size_t column) {
    CHECK_LT(row, 4);
    CHECK_LT(column, 4);
    return data_[row * 4 + column];
  }

  T operator[](size_t row, size_t column) const {
    CHECK_LT(row, 4);
    CHECK_LT(column, 4);
    return data_[row * 4 + column];
  }

  auto Load(const auto& delegate) const {
    return HWY_STATIC_DISPATCH(LoadImpl<T>)(*this, delegate);
  }

  template <typename D, typename F>
    requires std::convertible_to<T, typename D::T> &&
             std::invocable<F, D, typename hn::VFromD<D>,
                            typename hn::VFromD<D>, typename hn::VFromD<D>,
                            typename hn::VFromD<D>>
  auto LoadHighway(D d, const F& delegate) const {
    const T* HWY_RESTRICT data = data_.data();
    return delegate(d, hn::LoadU(d, data), hn::LoadU(d, data + 4),
                    hn::LoadU(d, data + 8), hn::LoadU(d, data + 12));
  }

  template <typename D>
    requires std::convertible_to<T, typename D::T>
  void Store(D d, const hn::VFromD<D>& v0, const hn::VFromD<D>& v1,
             const hn::VFromD<D>& v2, const hn::VFromD<D>& v3) {
    T* HWY_RESTRICT it = data_.data();
    hn::StoreU(v0, d, it);
    hn::StoreU(v1, d, it + 4);
    hn::StoreU(v2, d, it + 8);
    hn::StoreU(v3, d, it + 12);
  }

  template <typename S>
  friend void AbslStringify(S& s, const Tile4x4& layout) {
    auto data = layout.data_;
    s.Append("{\n");
    s.Append(absl::StrJoin({data.subspan(0, 4), data.subspan(4, 4),
                            data.subspan(8, 4), data.subspan(12, 4)},
                           ",\n", [](std::string* s, std::span<const T> data) {
                             *s = absl::StrCat(
                                 *s, "  { ",
                                 absl::StrJoin(data, ", ",
                                               [](std::string* ss, const T& n) {
                                                 *ss = absl::StrCat(*ss, n);
                                               }),
                                 " }");
                           }));
    s.Append("\n}");
  }

  friend std::ostream& operator<<(std::ostream& stream, const Tile4x4& tile) {
    stream << absl::StrCat(tile);
    return stream;
  }

 private:
  void Assign(const auto& other) {
    HWY_STATIC_DISPATCH(LoadImpl<T>)(
        other, [&](auto d, auto v0, auto v1, auto v2, auto v3) {
          Store(d, v0, v1, v2, v3);
        });
  }

  std::span<T, 16> data_;
};

template <typename T1>
ops::ScalarMulOp<T1> operator*(const T1& a, typename T1::value_type b) {
  return ops::ScalarMulOp{a, b};
}

template <typename T1>
ops::ScalarMulOp<T1> operator*(typename T1::value_type b, const T1& a) {
  return ops::ScalarMulOp{a, b};
}

template <typename A, typename B>
auto operator+(const A& a, const B& b) {
  return ops::AddOp{a, b};
}

template <typename T>
ops::MulOp<T> operator*(Tile4x4<T> a, Tile4x4<T> b) {
  return ops::MulOp<T>{std::move(a), std::move(b)};
}

template <typename Tile>
auto Transpose(Tile&& tile) {
  return ops::TransposeOp{std::forward<Tile>(tile)};
}

using FloatTile4x4 = Tile4x4<float>;

}  // namespace uchen::tensor

#endif  // UCHEN_TENSOR_TILE_H