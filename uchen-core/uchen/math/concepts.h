#ifndef UCHEN_MATH_CONCEPTS_H
#define UCHEN_MATH_CONCEPTS_H

#include <concepts>
#include <cstdint>

#include "uchen/math/matrix_core.h"

namespace uchen::math {

template <typename Store>
concept MatrixStore = requires {
  { Store::R } -> std::convertible_to<uint32_t>;
  { Store::C } -> std::convertible_to<uint32_t>;
} && std::derived_from<Store, MatrixBase> && requires(const Store store) {
  { store.GetColumnMajor(0) } -> std::convertible_to<float>;
  { store.begin() } -> std::input_or_output_iterator;
  { store.end() } -> std::input_or_output_iterator;
  { store.begin_across() } -> std::input_or_output_iterator;
  { store.end_across() } -> std::input_or_output_iterator;
};

template <typename M, uint32_t R, uint32_t C>
concept MatrixDim = std::derived_from<M, MatrixWithSize<R, C>>;

template <typename S>
concept RowContinuous = requires(const S s) {
  { s.begin() } -> std::contiguous_iterator;
};

template <typename S>
concept ColumnContinuous = requires(const S s) {
  { s.begin_across() } -> std::contiguous_iterator;
};

}  // namespace uchen::math

#endif  // UCHEN_MATH_CONCEPTS_H