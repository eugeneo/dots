#ifndef UCHEN_TEXT_EMBEDDINGS_TRAINING_H
#define UCHEN_TEXT_EMBEDDINGS_TRAINING_H

#include <cstddef>

#include "uchen/model.h"
#include "uchen/text/embeddings.h"

namespace uchen::text::impl {

template <size_t CS, size_t TT, size_t EL>
Vector<float, 1> ComputeGradients(
    const EmbeddingsLayer<CS, TT, EL>& /* layer */,
    const typename EmbeddingsLayer<CS, TT, EL>::input_t& input,
    Vector<float, CS * EL> output_gradients,
    const Parameters<TT * EL>& /* parameters */,
    std::span<float, TT * EL> parameter_gradients, const void* /* area */) {
  for (size_t i = 0; i < CS; ++i) {
    for (size_t j = 0; j < EL; ++j) {
      parameter_gradients[input[i] * EL + j] += output_gradients[i * EL + j];
    }
  }
  return {};
}

}  // namespace uchen::text::impl

#endif  // EMBEDDINGS_TRAINING_H