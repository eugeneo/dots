#ifndef UCHEN_TEXT_ATTENTION_H
#define UCHEN_TEXT_ATTENTION_H

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include "uchen/layer_traits.h"
#include "uchen/math/matrix.h"
#include "uchen/math/softmax.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/text/embeddings.h"

namespace uchen {
namespace text {
namespace impl {

constexpr size_t PC(size_t W) { return W * W * 3; }

template <size_t Ts, size_t W>
class AttentionLayerContext {
 public:
  using parameters = Parameters<PC(W)>;

  const math::RowMajorMatrix<Ts, W>& Calculate(
      const math::RowMajorView<Ts, W>& input, const parameters& parameters) {
    std::span s = parameters;
    // Computing the keys, queries and values
    //
    // Weights are transposed. They come from the parameter store so the order
    // there does not matter. Pretransposing them allows for more efficient
    // matrix multiplication. Cache lines and such.
    //
    auto keys_params = math::AsColumnMajorView<W, W>(s);
    auto queries_params =
        math::AsColumnMajorView<W, W>(s.template subspan<W * W, W * W>());
    auto values_params =
        math::AsColumnMajorView<W, W>(s.template subspan<2 * W * W, W * W>());
    keys_ = input * keys_params;
    queries_ = input * queries_params;
    values_ = input * values_params;
    attention_ = keys_.transposed() * queries_;
    attention_softmax_ = math::Softmax(attention_);
    output_matrix_ = values_ * attention_softmax_;
    return output_matrix_;
  }

  const auto& keys() const { return keys_; }
  const auto& queries() const { return queries_; }
  const auto& values() const { return values_; }
  const auto& attention() const { return attention_; }
  const auto& attention_softmax() const { return attention_softmax_; }

 private:
  math::ColumnMajorMatrix<Ts, W> keys_;
  math::ColumnMajorMatrix<Ts, W> queries_;
  math::RowMajorMatrix<Ts, W> values_;
  math::ColumnMajorMatrix<W, W> attention_;
  math::ColumnMajorMatrix<W, W> attention_softmax_;
  math::RowMajorMatrix<Ts, W> output_matrix_;
};

template <size_t Ts, size_t W>
struct AttentionLayer {
  using input_t = Vector<float, Ts * W>;

  const math::RowMajorMatrix<Ts, W>& operator()(
      const input_t& input, const Parameters<PC(W)>& parameters,
      memory::LayerContext<AttentionLayerContext<Ts, W>>* context) const {
    return context->GetScratchArea()->Calculate(input, parameters);
  }
};

template <size_t Ts, size_t W>
Parameters<PC(W)> ParameterProvider(const AttentionLayer<Ts, W>& layer,
                                    std::span<const float> parameters,
                                    std::shared_ptr<memory::Deletable> ref) {
  return {parameters, std::move(ref)};
}

struct AttentionDescriptor {
 private:
  template <typename L>
  struct MatrixProps;

  template <size_t CS, size_t TT, size_t EL>
  struct MatrixProps<EmbeddingsLayer<CS, TT, EL>> {
    static constexpr size_t kContextSize = CS;
    static constexpr size_t kTokenTypes = TT;
    static constexpr size_t kEmbeddingLength = EL;
  };

 public:
  template <typename... Ls>
  constexpr auto stack(const Model<Ls...>& model) const {
    using M = std::remove_cvref_t<decltype(model)>;
    constexpr size_t last = M::kLayers - 1;
    using Props = MatrixProps<typename M::template L<last>>;
    return AttentionLayer<Props::kContextSize, Props::kEmbeddingLength>();
  }
};
}  // namespace impl

constexpr Layer<impl::AttentionDescriptor> Attention;

}  // namespace text

template <size_t Ts, size_t W, typename I>
struct LayerTraits<text::impl::AttentionLayer<Ts, W>, I>
    : public LayerTraitFields<Vector<float, Ts * W>, text::impl::PC(W),
                              text::impl::AttentionLayerContext<Ts, W>> {};

}  // namespace uchen

#endif  // UCHEN_TEXT_ATTENTION_H