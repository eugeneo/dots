#ifndef UCHEN_TRAINING_MODEL_GRADIENTS_H
#define UCHEN_TRAINING_MODEL_GRADIENTS_H

#include <array>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"

#include "uchen/parameters.h"
#include "uchen/training/backprop.h"
#include "uchen/training/parameter_gradients.h"

namespace uchen::training {

template <typename V>
struct Materializer;

template <typename V, size_t E>
struct Materializer<Vector<V, E>> {
  static Vector<V, E> materialize(memory::ArrayStore<V, E>* store) {
    return Vector<V, E>(store->data());
  }

  static Vector<float, E> materialize(std::array<float, E>* store) {
    return Vector<float, E>(*store, nullptr);
  }
};

template <typename V>
struct ConcreteTypeForGradient {
  using type = V;
};

template <typename M, typename I>
class ForwardPassResult {
 public:
  ForwardPassResult(const M* m, const I& input,
                    const ModelParameters<M>& parameters)
      : model_(m),
        input_(input),
        parameters_(parameters),
        ctx_(std::make_unique<ContextForGradientCalculation>()) {
    result_ = (*m)(input, parameters, *ctx_);
  }

  typename M::output_t result() const { return result_; }

  auto CalculateParameterGradients(
      const Vector<float, M::output_t::elements>& loss_gradients) {
    ParameterGradients<M> gradients;
    auto input_gradients =
        ComputeParameterGradients(loss_gradients, gradients, M::kLayerIndexes);
    return std::make_pair(std::move(input_gradients), std::move(gradients));
  }

 private:
  template <size_t L>
  class LayerBackpropagation {
   public:
    LayerBackpropagation(ForwardPassResult<M, I>* fpr,
                         ParameterGradients<M>* gradients)
        : fpr_(fpr), gradients_(gradients) {}

    const auto layer_input() {
      if constexpr (L == 0) {
        return fpr_->input_;
      } else if constexpr (M::template Traits<L - 1>::skip) {
        if constexpr (L == 1) {
          return fpr_->input_;
        } else {
          return Materializer<typename M::template Traits<L - 2>::output_t>::
              materialize(&std::get<L - 2>(fpr_->ctx_->tuple()));
        }
      } else {
        return Materializer<typename M::template Traits<L - 1>::output_t>::
            materialize(&std::get<L - 1>(fpr_->ctx_->tuple()));
      }
    }

    const auto layer_output() {
      if constexpr (L == M::kLayers - 1) {
        return fpr_->result_;
      } else {
        return Materializer<typename M::template Traits<L>::output_t>::
            materialize(&std::get<L>(fpr_->ctx_->tuple()));
      }
    }

    template <size_t C>
    auto operator<(const Vector<float, C>& loss_grad) {
      auto&& layer_in = layer_input();
      auto&& params = fpr_->parameters_.template layer_parameters<L>();
      auto&& grads = gradients_->template layer_parameter_gradients<L>();
      auto&& ctx = std::get<L>(fpr_->ctx_->tuple());

      if constexpr (requires {
                      ComputeGradients(fpr_->model_->template layer<L>(),
                                       layer_in, loss_grad, params, grads,
                                       &ctx);
                    }) {
        // Old overload
        return ComputeGradients(fpr_->model_->template layer<L>(), layer_in,
                                loss_grad, params, grads, &ctx);
      } else {
        // New overload
        return ComputeGradients(fpr_->model_->template layer<L>(), layer_in,
                                loss_grad, params, grads, &ctx, layer_output());
      }
    }

   private:
    ForwardPassResult<M, I>* fpr_;
    ParameterGradients<M>* gradients_;
  };

 public:
  class ContextForGradientCalculation final : public memory::Context<M, I> {
   public:
    ContextForGradientCalculation()
        : vtable_(BuildVtable(&areas_, M::kLayerIndexes)) {}

    typename memory::Context<M, I>::vtable_t& GetLayerArenas() override {
      return vtable_;
    }

    auto& tuple() { return areas_; }

   private:
    template <typename T, size_t... Idx>
    static auto BuildVtable(T* tuple,
                            std::index_sequence<Idx...> /* ignore */) {
      return std::make_tuple(Getter<T, Idx>(tuple)...);
    }

    template <typename T, size_t Idx>
    static absl::AnyInvocable<
        typename M::template Traits<Idx>::scratch_area_t*()>
    Getter(T* tuple) {
      return [=]() { return &std::get<Idx>(*tuple); };
    }

    template <size_t... Ind>
    static auto LayerAreasTuple(std::index_sequence<Ind...> /* unused */) {
      return std::tuple<typename ConcreteTypeForGradient<
          typename M::template Traits<Ind>::scratch_area_t>::type...>();
    }

    typename memory::Context<M, I>::vtable_t vtable_;
    decltype(LayerAreasTuple(M::kLayerIndexes)) areas_;
  };

  template <size_t... Layer>
  auto ComputeParameterGradients(
      const Vector<float, M::output_t::elements>& loss_gradients,
      ParameterGradients<M>& gradients,
      std::index_sequence<Layer...> /* unused */) {
    return (LayerBackpropagation<Layer>(this, &gradients) < ... <
            loss_gradients);
  }

  const M* model_;
  const I input_;
  const ModelParameters<M> parameters_;
  std::unique_ptr<ContextForGradientCalculation> ctx_;
  typename M::output_t result_;
};

}  // namespace uchen::training

#endif  // UCHEN_TRAINING_MODEL_GRADIENTS_H