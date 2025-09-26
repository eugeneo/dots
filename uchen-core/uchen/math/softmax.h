#ifndef UCHEN_MATH_SOFTMAX_H
#define UCHEN_MATH_SOFTMAX_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include "uchen/math/concepts.h"
#include "uchen/math/matrix.h"
#include "uchen/math/primitives.h"

namespace uchen::math {
namespace details {

template <ColumnContinuous S1, ColumnContinuous S2>
void Softmax(const Matrix<S1>& input, Matrix<S2>& output) {
  std::span input_span{std::to_address(input.begin_across()), input.size()};
  std::span output_span{std::to_address(output.begin_across()), output.size()};
  ColumnWiseSoftmax(input_span, output_span, S1::R);
}

template <RowContinuous S1, RowContinuous S2>
void Softmax(const Matrix<S1>& input, Matrix<S2>& output) {
  std::span input_span{std::to_address(input.begin()), input.size()};
  std::span output_span{std::to_address(output.begin()), output.size()};
  RowWiseSoftmax(input_span, output_span, S1::C);
}

template <typename S1, typename S2>
void Softmax(const Matrix<S1>& input, Matrix<S2>& output) {
  for (uint32_t c = 0; c < S1::C; ++c) {
    float max = std::numeric_limits<float>::min();
    auto start = input.begin_across() + c * S1::R;
    auto end = start + S1::R;
    for (auto it = start; it != end; ++it) {
      max = std::max(max, *it);
    }
    auto it2 = output.begin_across() + c * S1::R;
    float exp_accum = 0;
    for (auto it = start; it != end; ++it, ++it2) {
      float v = std::exp(*it - max);
      exp_accum += v;
      *it2 = v;
    }
    it2 = output.begin_across() + c * S1::R;
    for (auto it = start; it != end; ++it, ++it2) {
      *it2 /= exp_accum;
    }
  }
}

}  // namespace details

template <typename S1>
auto Softmax(const Matrix<S1>& input) {
  return [&](MatrixDim<S1::R, S1::C> auto& output) {
    details::Softmax(input, output);
  };
}

}  // namespace uchen::math

#endif  // UCHEN_MATH_SOFTMAX_H