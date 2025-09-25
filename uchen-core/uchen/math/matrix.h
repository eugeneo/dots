#ifndef UCHEN_MATH_MATRIX_H
#define UCHEN_MATH_MATRIX_H

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <span>
#include <utility>

#include "absl/log/log.h"  // IWYU pragma: keep
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "uchen/math/concepts.h"
#include "uchen/math/matrix_core.h"
#include "uchen/math/matrix_functional.h"
#include "uchen/math/multiplication.h"

namespace uchen::math {

template <MatrixStore Store>
class Matrix final : public Store {
 public:
  static constexpr uint32_t Rows = Store::R;
  static constexpr uint32_t Columns = Store::C;

  using value_type = float;

  Matrix(const MatrixDim<Rows, Columns> auto& other) { CopyFrom(other); }

  constexpr Matrix(const std::invocable<Matrix&> auto& fn) {
    std::invoke(fn, *this);
  }

  template <typename... Args>
  explicit constexpr Matrix(Args&&... args)
      : Store(std::forward<Args>(args)...) {}

  template <typename S>
  Matrix& operator=(const Matrix<S>& other) {
    CopyFrom(other);
    return *this;
  }

  Matrix& operator=(const std::invocable<Matrix&> auto& fn) {
    std::invoke(fn, *this);
    return *this;
  }

  template <typename S>
  Matrix& operator+=(const Matrix<S>& other) {
    auto begin = this->OtherBeginCompatible(other);
    for (auto it = this->begin_natural(); it != this->end_natural(); ++it) {
      *it += *begin++;
    }
    return *this;
  }

  template <typename S>
  Matrix& operator-=(const Matrix<S>& other) {
    auto begin = this->OtherBeginCompatible(other);
    for (auto it = this->begin_natural(); it != this->end_natural(); ++it) {
      *it -= *begin++;
    }
    return *this;
  }

  Matrix& operator*=(float f) {
    for (auto it = this->begin_natural(); it != this->end_natural(); ++it) {
      *it *= f;
    }
    return *this;
  }

  Matrix& operator/=(float f) {
    for (auto it = this->begin_natural(); it != this->end_natural(); ++it) {
      *it /= f;
    }
    return *this;
  }

  template <typename S>
  bool operator==(const Matrix<S>& other) const {
    return std::equal(this->begin(), this->end(), other.begin());
  }

  auto transposed() {
    return Matrix<typename Transposed<Store>::type>(this->data_view());
  }

 private:
  void CopyFrom(const MatrixDim<Rows, Columns> auto& other) {
    std::copy(this->OtherBeginCompatible(other),
              this->OtherEndCompatible(other), this->begin_natural());
  }
};

template <uint32_t R, uint32_t C>
using RowMajorMatrix =
    Matrix<RowMajorLayout<R, C, std::array<float, static_cast<size_t>(R) * C>>>;
template <uint32_t R, uint32_t C>
using ColumnMajorMatrix = Matrix<
    ColumnMajorLayout<R, C, std::array<float, static_cast<size_t>(R) * C>>>;
template <uint32_t R, uint32_t C>
using RowMajorView = Matrix<
    RowMajorLayout<R, C, std::span<const float, static_cast<size_t>(R) * C>>>;

template <uint32_t R, uint32_t C>
auto FnMatrix(absl::AnyInvocable<float(uint32_t, uint32_t) const> fn) {
  return Matrix<details::TwoArgFn<R, C>>(std::move(fn));
}

template <uint32_t R, uint32_t C>
auto FnMatrix(absl::AnyInvocable<float(size_t) const> fn) {
  return Matrix<details::SingleArgFn<R, C>>(std::move(fn));
}

template <uint32_t R, uint32_t C = 1>
auto AsColumnMajorView(std::array<float, static_cast<size_t>(R) * C>& data) {
  return Matrix<
      ColumnMajorLayout<R, C, std::span<float, static_cast<size_t>(R) * C>>>(
      data);
}

template <uint32_t R, uint32_t C = 1>
auto AsColumnMajorView(std::span<float> data) {
  return Matrix<
      ColumnMajorLayout<R, C, std::span<float, static_cast<size_t>(R) * C>>>(
      data);
}

template <uint32_t R, uint32_t C = 1>
auto AsColumnMajorView(std::span<const float> data) {
  return Matrix<ColumnMajorLayout<R, C, std::span<const float, (R)*C>>>(data);
  ;
}

template <uint32_t R, uint32_t C = 1>
auto AsRowMajorView(std::array<float, static_cast<size_t>(R) * C>& data) {
  constexpr size_t Size = static_cast<size_t>(R) * C;
  return Matrix<RowMajorLayout<R, C, std::span<float, Size>>>(
      std::span<float, Size>(data));
}

template <uint32_t R, uint32_t C = 1>
auto AsRowMajorView(std::span<float> data) {
  return Matrix<
      RowMajorLayout<R, C, std::span<float, static_cast<size_t>(R) * C>>>(data);
}

template <uint32_t R, uint32_t C = 1>
auto AsRowMajorView(std::span<const float> data) {
  return Matrix<
      RowMajorLayout<R, C, std::span<const float, static_cast<size_t>(R) * C>>>(
      data);
}

template <typename S1, typename S2>
auto operator+(const Matrix<S1>& a, const Matrix<S2>& b) {
  return FnMatrix<S1::R, S1::C>(
      [&](size_t i) { return a.GetRowMajor(i) + b.GetRowMajor(i); });
}

template <typename S1, typename S2>
auto operator*(const Matrix<S1>& a, const Matrix<S2>& b) {
  return mult::Multiplication(&a, &b);
}

template <typename S, typename Op>
auto operator+(const Op& op, const Matrix<S>& m) {
  return [&]<typename S2>(Matrix<S2>& result) {
    std::invoke(op, result);
    result += m;
  };
}

void AbslStringify(auto& sink, const MatrixBase& p) {
  constexpr size_t MaxElements = 15;
  absl::Format(&sink, "(%ld,%ld){", p.rows(), p.columns());
  for (size_t i = 0;
       i < std::min(static_cast<size_t>(p.rows()) * p.columns(), MaxElements);
       ++i) {
    if (i > 0) {
      absl::Format(&sink, ",");
    }
    absl::Format(&sink, "%g", p.GetRowMajor(i));
  }
  if (p.rows() * p.columns() > MaxElements) {
    absl::Format(&sink, ",...");
  }
  absl::Format(&sink, "}");
}

template <typename S>
std::ostream& operator<<(std::ostream& stream, const Matrix<S>& matrix) {
  stream << absl::StrCat(matrix);
  return stream;
}

}  // namespace uchen::math

#endif  // UCHEN_MATH_MATRIX_H