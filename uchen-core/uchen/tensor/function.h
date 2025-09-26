#ifndef UCHEN_TENSOR_FUNCTION_H
#define UCHEN_TENSOR_FUNCTION_H

#include <cstddef>

#include "absl/log/log.h"  // IWYU pragma: keep

#include "uchen/tensor/float_tensor.h"

namespace uchen::core {

inline TensorProjection Transpose(
    const BasicTensor& input ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  return TensorProjection(input, details::TransposeTranslator());
}

}  // namespace uchen::core

#endif  // UCHEN_TENSOR_FUNCTION_H