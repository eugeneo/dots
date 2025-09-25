#ifndef UCHEN_RNN_H
#define UCHEN_RNN_H

#include <stddef.h>

#include <array>
#include <memory>
#include <type_traits>

#include "uchen/inferrence_context.h"
#include "uchen/layer_traits.h"
#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/memory.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/vector.h"

namespace uchen {

// Not sure these need to be exposed.
namespace internal {

template <typename T>
struct result_type {
  using type = typename T::value_type;
};

template <typename Stream>
using result_t = typename result_type<Stream>::type;

template <typename I, typename M, size_t HiddenState, typename NonLinearity>
constexpr size_t RnnParameters() {
  return M::all_parameters_count() +
         Model<Linear<typename M::output_t, HiddenState>,
               NonLinearity>::all_parameters_count();
}

template <typename I, typename M, size_t HS, typename HH>
  requires(result_t<I>::elements > 0)
class RnnScratchArea {
 public:
  using mm_input_t =
      Vector<typename I::value_type::value_type, HS + I::value_type::elements>;
  using value_type = typename result_t<I>::value_type;

  virtual ~RnnScratchArea() = default;

  virtual Vector<value_type, HS> hh_run(
      const typename M::output_t& mm_output, const HH& model,
      const ModelParameters<HH>& parameters) = 0;
  virtual typename M::output_t mm_run(const mm_input_t& mm_input,
                                      const M& model,
                                      const ModelParameters<M>& parameters) = 0;
  virtual std::span<value_type, HS + result_t<I>::elements>
  get_input_store() = 0;
};

template <typename I, typename M, size_t HS, typename HH>
class RnnScratchAreaForInferrence : public RnnScratchArea<I, M, HS, HH> {
 private:
  using Base = RnnScratchArea<I, M, HS, HH>;

 public:
  Vector<typename Base::value_type, HS> hh_run(
      const typename M::output_t& input, const HH& model,
      const ModelParameters<HH>& parameters) override {
    return model(input, parameters, hh_context_);
  }

  typename M::output_t mm_run(const typename Base::mm_input_t& input,
                              const M& model,
                              const ModelParameters<M>& parameters) override {
    return model(input, parameters, mm_context_);
  }

  std::span<typename Base::value_type, HS + result_t<I>::elements>
  get_input_store() override {
    return input_store_;
  }

 private:
  std::array<typename Base::mm_input_t::value_type, Base::mm_input_t::elements>
      input_store_;
  ContextForInfer<M, typename Base::mm_input_t> mm_context_;
  ContextForInfer<HH, typename M::output_t> hh_context_;
};

}  // namespace internal

template <typename Model, typename MI, typename I, typename M, size_t HS,
          typename HH>
struct ConcreteType<ContextForInfer<Model, MI>,
                    internal::RnnScratchArea<I, M, HS, HH>> {
  using type = internal::RnnScratchAreaForInferrence<I, M, HS, HH>;
};

template <
    typename I, typename M, size_t HiddenState,
    typename NonLinearity = ElementWise<
        Vector<typename internal::result_t<I>::value_type, HiddenState>, Relu>>
class RnnLayer {
 public:
  using input_t = I;
  using output_t = typename M::output_t;
  static constexpr size_t parameter_count =
      internal::RnnParameters<I, M, HiddenState, NonLinearity>();
  template <typename In>
  using VectorWithHiddenState =
      Vector<typename M::output_t::value_type,
             HiddenState + std::remove_reference_t<
                               decltype(*std::declval<I>().begin())>::elements>;
  using HH = Model<Linear<output_t, HiddenState>, NonLinearity>;
  using ScratchArea = internal::RnnScratchArea<I, M, HiddenState, HH>;

  constexpr RnnLayer(const M& model) : model_(model) {}

  output_t operator()(const input_t& input,
                      const Parameters<parameter_count>& parameters,
                      memory::LayerContext<ScratchArea>* context) const {
    // model -> hh_model -> model -> hh_model -> ... -> model
    std::unique_ptr<ScratchArea> handle;
    CHECK_NE(context, nullptr);
    Vector<typename M::input_t::value_type, HiddenState> state(0);
    output_t result(0);
    constexpr size_t par_count = M::all_parameters_count();
    bool first = true;
    ScratchArea* scratch_area = context->GetScratchArea();
    ModelParameters mm_parameters =
        ModelParameters(&model_, parameters.template starting<0, par_count>());
    ModelParameters hh_parameters =
        ModelParameters(&hh_model_, parameters.template starting<par_count>());
    for (const auto& token : input) {
      if (!first) {
        state = scratch_area->hh_run(result, hh_model_, hh_parameters);
      }
      first = false;
      result = scratch_area->mm_run(
          token.Join(state, scratch_area->get_input_store()), model_,
          mm_parameters);
    }
    return result;
  }

  const M* model() const { return &model_; }
  const auto* hh_model() const { return &hh_model_; }

 private:
  const M model_;
  HH hh_model_;
};

template <typename I, typename M, size_t HS, typename NL>
Parameters<internal::RnnParameters<I, M, HS, NL>()> ParameterProvider(
    const RnnLayer<I, M, HS, NL>& layer, std::span<const float> buffer,
    std::shared_ptr<memory::Deletable> handle) {
  return Parameters<internal::RnnParameters<I, M, HS, NL>()>(buffer,
                                                             std::move(handle));
}

template <typename I, typename M, size_t HiddenState, typename NonLinearity>
struct LayerTraits<RnnLayer<I, M, HiddenState, NonLinearity>, I>
    : public LayerTraitFields<
          typename M::output_t,
          internal::RnnParameters<I, M, HiddenState, NonLinearity>(),
          internal::RnnScratchArea<
              I, M, HiddenState,
              typename RnnLayer<I, M, HiddenState, NonLinearity>::HH>> {};

namespace layers {
template <typename In, size_t HiddenState>
constexpr auto Rnn(const auto& nested) {
  using I = typename internal::result_t<In>::template resize<HiddenState>;
  auto m = Input<I> | nested;
  return Model<RnnLayer<In, decltype(m), HiddenState>>(
      RnnLayer<In, decltype(m), HiddenState>(m));
}

}  // namespace layers

}  // namespace uchen
#endif  // UCHEN_RNN_H