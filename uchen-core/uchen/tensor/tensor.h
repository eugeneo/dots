#ifndef UCHEN_TENSOR_TENSOR_H_
#define UCHEN_TENSOR_TENSOR_H_

#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"

namespace uchen::tensor {

class SimpleLayout {
 public:
  constexpr SimpleLayout(size_t rows, size_t columns)
      : rows_(rows), columns_(columns) {}

  constexpr size_t matrix_elements() const { return rows_ * columns_; }

  constexpr std::pair<size_t, size_t> dims() const { return {rows_, columns_}; }

 protected:
  size_t rows_ = 0;
  size_t columns_ = 0;
};

class RowMajor : private SimpleLayout {
 public:
  template <typename S>
  friend void AbslStringify(S& s, const RowMajor& /* unused */) {
    s.Append("RowMajor");
  }

  constexpr RowMajor(size_t rows, size_t columns)
      : SimpleLayout(rows, columns) {}

  constexpr std::pair<size_t, size_t> FromArrayIndex(size_t index) const {
    return {index / columns_, index % columns_};
  }

  constexpr size_t ToArrayIndex(size_t row, size_t column) const {
    return column + row * columns_;
  }

  using SimpleLayout::dims;
  using SimpleLayout::matrix_elements;
};

class ColumnMajor : private SimpleLayout {
 public:
  template <typename S>
  friend void AbslStringify(S& s, const ColumnMajor& /* unused */) {
    s.Append("ColumnMajor");
  }

  constexpr ColumnMajor(size_t rows, size_t columns)
      : SimpleLayout(rows, columns) {}

  constexpr std::pair<size_t, size_t> FromArrayIndex(size_t index) const {
    return {index % rows_, index / rows_};
  }

  constexpr size_t ToArrayIndex(size_t row, size_t column) const {
    return column * rows_ + row;
  }

  using SimpleLayout::dims;
  using SimpleLayout::matrix_elements;
};

template <typename L>
constexpr size_t IndexFromTo(const L& /* from */, const L& /* to */, size_t i) {
  return i;
}

constexpr size_t IndexFromTo(const auto& from, const auto& to, size_t i) {
  CHECK_EQ(from.dims().first, to.dims().first);
  CHECK_EQ(from.dims().second, to.dims().second);
  size_t index = i % from.matrix_elements();
  const auto& [row, column] = from.FromArrayIndex(index);
  return i - index + to.ToArrayIndex(row, column);
}

class Layout {
 public:
  struct MemoryLayoutInfo {
    size_t fast_dim_size;
    size_t slow_dim_size;
  };

  static constexpr struct ColumnMajor_t {
    static constexpr std::string_view kLabel = "ColumnMajor";
    template <size_t... Ds>
    static constexpr size_t elements = (1 * ... * Ds);
  } column_major;

  static constexpr struct RowMajor_t {
    static constexpr std::string_view kLabel = "RowMajor";
    template <size_t... Ds>
    static constexpr size_t elements = (1 * ... * Ds);
  } row_major;

  constexpr Layout(ColumnMajor_t /* unused */, size_t rows, size_t columns)
      : matrix_layout_(ColumnMajor(rows, columns)) {}
  constexpr Layout(RowMajor_t /* unused */, size_t rows, size_t columns)
      : matrix_layout_(RowMajor(rows, columns)) {}

  template <typename V>
  auto visit(V&& visitor) const {
    return std::visit(std::forward<V>(visitor), matrix_layout_);
  }

  size_t ArrayIndex(absl::Span<const size_t> dimensions,
                    absl::Span<const size_t> index) const;

  template <size_t C>
    requires(C >= 2)
  std::array<size_t, C> Index(const std::array<size_t, C>& dims,
                              size_t index) const {
    std::array<size_t, C> result;
    size_t matrix_index = index % (dims[C - 2] * dims.back());
    std::tie(result[C - 2], result.back()) = visit([&](const auto& layout) {
      return layout.FromArrayIndex(matrix_index);
    });
    size_t matrix_id = index / (dims[C - 2] * dims.back());
    for (size_t i = C - 2; i > 0; --i) {
      result[i - 1] = matrix_id % dims[i - 1];
      matrix_id /= dims[i - 1];
    }
    return result;
  }

  MemoryLayoutInfo memory_layout_info() const;

  template <typename S>
  friend void AbslStringify(S& s, const Layout& layout) {
    auto mem = layout.memory_layout_info();
    s.Append(absl::Substitute("Fast: $0 slow: $1", mem.fast_dim_size,
                              mem.slow_dim_size));
  }

 private:
  std::variant<RowMajor, ColumnMajor> matrix_layout_;
};

template <typename T, size_t D, size_t... Ds>
  requires((D > 0 && (Ds > 0)) && ...)
class TensorRef {
 public:
  static constexpr std::array Dims = {D, Ds...};

  explicit TensorRef(std::span<T> data, const auto& layout_tag)
      : data_(data), layout_(layout_tag, Dims[Dims.size() - 2], Dims.back()) {}

  Layout layout() const { return layout_; }

  std::span<const T> data() const { return data_; }

 private:
  std::span<const T> data_;
  Layout layout_;
};

template <typename T, size_t D, size_t... Ds>
  requires((D > 0 && (Ds > 0)) && ...)
class TensorView;

template <typename T, size_t D, size_t... Ds>
class TensorOperation {
 public:
  virtual ~TensorOperation() = default;
  virtual void Apply(TensorView<T, D, Ds...>& tensor) const = 0;
};

template <typename T, size_t D, size_t... Ds>
  requires((D > 0 && (Ds > 0)) && ...)
class TensorView {
 public:
  static constexpr std::array Dims = {D, Ds...};

  TensorView(std::span<T> data, const auto& layout_tag)
      : data_(data), layout_(layout_tag, Dims[Dims.size() - 2], Dims.back()) {}

  TensorView& operator=(const TensorOperation<T, D, Ds...>& operation) {
    operation.Apply(*this);
    return *this;
  }

  Layout layout() const { return layout_; }
  std::span<T> data() { return data_; }
  std::span<const T> data() const { return data_; }

 private:
  std::span<T> data_;
  Layout layout_;
};

}  // namespace uchen::tensor

#endif  // ndef UCHEN_TENSOR_TENSOR_H_