#ifndef TEST_TENSOR_TENSOR_TEST_UTIL_H
#define TEST_TENSOR_TENSOR_TEST_UTIL_H

#include "uchen/tensor/float_tensor.h"

namespace uchen::core::testing {

template <size_t... Dims>
constexpr bool IncrementIndex(std::span<size_t, sizeof...(Dims)> index,
                              size_t dim = sizeof...(Dims) - 1) {
  index[dim] += 1;
  if (index[dim] == std::array{Dims...}[dim]) {
    index[dim] = 0;
    if (dim == 0) {
      return false;
    }
    return IncrementIndex<Dims...>(index, dim - 1);
  }
  return true;
}

template <size_t... Dims>
  requires(sizeof...(Dims) > 0)
constexpr FloatTensor<Dims...> IotaTensor() {
  FloatTensor<Dims...> result;
  constexpr std::array kDims = {Dims...};
  std::array<size_t, sizeof...(Dims)> index;
  index.fill(0);
  do {
    float v = index[0];
    for (size_t i = 1; i < sizeof...(Dims); ++i) {
      v = v * kDims[i] + index[i];
    }
    result(index) = v + 1;
  } while (IncrementIndex<Dims...>(index));
  return result;
}

template <size_t... Dims>
  requires(sizeof...(Dims) > 0)
FloatTensor<Dims...> Generate(const auto& func) {
  FloatTensor<Dims...> result;
  std::array<size_t, sizeof...(Dims)> index;
  index.fill(0);
  do {
    result(index) = std::invoke(func, index);
  } while (IncrementIndex<Dims...>(index));
  return result;
}

}  // namespace uchen::core

#endif  // TEST_TENSOR_TENSOR_TEST_UTIL_H