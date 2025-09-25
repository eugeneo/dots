#ifndef UCHEN_MATH_MULTIPLICATION_H
#define UCHEN_MATH_MULTIPLICATION_H

#include <concepts>
#include <cstdint>

#include "absl/log/log.h"  // IWYU pragma: keep

#include "uchen/math/concepts.h"
#include "uchen/math/primitives.h"

namespace uchen::math {

template <MatrixStore Store>
class Matrix;

namespace mult {

template <ColumnContinuous S>
void PerElement(Matrix<S>& m, std::invocable<uint32_t, int32_t> auto fn) {
  auto it = m.begin_across();
  for (uint32_t column = 0; column < S::C; ++column) {
    for (uint32_t row = 0; row < S::R; ++row) {
      *it++ = fn(row, column);
    }
  }
}

template <typename S>
void PerElement(Matrix<S>& m, std::invocable<uint32_t, int32_t> auto fn) {
  auto it = m.begin();
  for (uint32_t row = 0; row < S::R; ++row) {
    for (uint32_t column = 0; column < S::C; ++column) {
      *it++ = fn(row, column);
    }
  }
}

template <RowContinuous S1, ColumnContinuous S2>
void Multiply(const Matrix<S1>& a_, const Matrix<S2>& b_, auto& result) {
  constexpr size_t RC = S1::C;
  std::span<const float> a = std::span(std::to_address(a_.begin()), S1::size());
  std::span<const float> b =
      std::span(std::to_address(b_.begin_across()), S2::size());
  PerElement(result, [&](uint32_t row, uint32_t column) {
    return DotProduct(a.subspan(row * RC, RC), b.subspan(column * RC, RC));
  });
}

template <typename S1, typename S2>
void Multiply(const Matrix<S1>& a, const Matrix<S2>& b, auto& result) {
  constexpr size_t RC = S1::C;
  PerElement(result, [&](uint32_t r, uint32_t c) {
    float sum = 0;
    for (uint32_t rc = 0; rc < RC; ++rc) {
      sum += a.GetRowMajor(r * RC + rc) * b.GetColumnMajor(c * RC + rc);
    }
    return sum;
  });
}

template <ColumnContinuous S1, RowContinuous S2, ColumnContinuous S3>
void MultiplyBest(const Matrix<S1>& a_, const Matrix<S2>& b_, S3& result) {
  constexpr size_t R = S1::R;
  constexpr size_t RC = S1::C;
  constexpr size_t C = S2::C;
  std::span<const float> a =
      std::span(std::to_address(a_.begin_across()), S1::size());
  std::span<const float> b = std::span(std::to_address(b_.begin()), S2::size());
  std::span<float> out =
      std::span(std::to_address(result.begin_across()), result.size());
  std::fill(out.begin(), out.end(), 0.f);
  for (uint32_t c = 0; c < C; ++c) {
    MatrixByVector(a, b.subspan(c * RC, RC), out.subspan(c * R, R));
  }
}

template <typename S1, typename S2>
  requires(S1::C == S2::R)
class Multiplication {
 public:
  constexpr Multiplication(const Matrix<S1>* a, const Matrix<S2>* b)
      : a_(*a), b_(*b) {}

  template <typename S>
    requires(S::R == S1::R && S::C == S2::C)
  constexpr void operator()(Matrix<S>& result) const {
    if constexpr (ColumnContinuous<S1> && RowContinuous<S2> &&
                  ColumnContinuous<S>) {
      MultiplyBest(a_, b_, result);
    } else {
      Multiply(a_, b_, result);
    }
  }

 private:
  const Matrix<S1>& a_;
  const Matrix<S2>& b_;
};

}  // namespace mult
}  // namespace uchen::math

#endif