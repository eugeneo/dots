#ifndef UCHEN_TEXT_EMBEDDING_H
#define UCHEN_TEXT_EMBEDDING_H

#include <cstddef>

#include "uchen/model.h"
#include "uchen/vector.h"

namespace uchen {
namespace text {
namespace impl {

template <size_t ContextSize, size_t TokenTypes, size_t EmbeddingLength>
struct EmbeddingsLayer {
  using input_t = std::span<const uint32_t, ContextSize>;

  Vector<float, ContextSize * EmbeddingLength> operator()(
      const input_t& input,
      const Parameters<TokenTypes * EmbeddingLength>& parameters,
      memory::LayerContext<
          memory::ArrayStore<float, ContextSize * EmbeddingLength>>* context)
      const {
    auto data_start = context->GetScratchArea()->data().begin();
    auto parameters_begin = parameters.begin();
    for (size_t i = 0; i < ContextSize; ++i) {
      uint32_t token = input[i];
      DCHECK_LT(token, TokenTypes);
      std::copy(parameters_begin + token * EmbeddingLength,
                parameters_begin + (token + 1) * EmbeddingLength,
                data_start + i * EmbeddingLength);
    }
    return Vector<float, ContextSize * EmbeddingLength>(
        context->GetScratchArea()->data());
  }
};

template <size_t CS, size_t TT, size_t EL>
Parameters<TT * EL> ParameterProvider(
    const EmbeddingsLayer<CS, TT, EL>& layer, std::span<const float> data,
    std::shared_ptr<memory::Deletable> handle) {
  return Parameters<TT * EL>(data, std::move(handle));
}

}  // namespace impl

template <size_t CS, size_t TT, size_t EL>
constexpr Model<impl::EmbeddingsLayer<CS, TT, EL>> Embeddings;

}  // namespace text

template <size_t CS, size_t TT, size_t EL>
struct LayerTraits<text::impl::EmbeddingsLayer<CS, TT, EL>,
                   typename text::impl::EmbeddingsLayer<CS, TT, EL>::input_t>
    : public VectorOutputLayer<Vector<float, CS * EL>, TT * EL> {};

}  // namespace uchen

#endif  // UCHEN_TEXT_EMBEDDING_H
