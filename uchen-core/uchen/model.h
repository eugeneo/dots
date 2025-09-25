#ifndef UCHEN_STATIC_UTILS_H_
#define UCHEN_STATIC_UTILS_H_

#include <stddef.h>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "uchen/inferrence_context.h"
#include "uchen/layer_traits.h"
#include "uchen/math/matrix.h"
#include "uchen/memory.h"
#include "uchen/parameters.h"
#include "uchen/vector.h"

namespace uchen {

namespace internal {

template <typename SA>
class InferenceLayerContext final : public memory::LayerContext<SA> {
 public:
  explicit InferenceLayerContext(SA* area) : area_(area) {}
  SA* GetScratchArea() override { return area_; }

 private:
  SA* area_;
};

}  // namespace internal

namespace concepts {

template <typename M>
concept InputLayer = requires(M m) {
  typename M::input_t;
  typename M::output_t;
};

}  // namespace concepts

template <typename L, typename I, size_t P>
typename LayerTraits<L, typename L::input_t>::output_t InvokeLayer(
    const L* layer, const I& input, const Parameters<P>& parameters,
    typename memory::LayerContext<typename LayerTraits<L, I>::scratch_area_t>&
        context) {
  using Context = std::remove_reference_t<decltype(context)>;
  if constexpr (std::is_invocable_v<L, const typename L::input_t&, Context*>) {
    return (*layer)(input, &context);
  } else if constexpr (std::is_invocable_v<L, const typename L::input_t&,
                                           const Parameters<P>&, Context*>) {
    return (*layer)(input, parameters, &context);
  } else if constexpr (std::is_invocable_v<L, const typename L::input_t&,
                                           const Parameters<P>&>) {
    return (*layer)(input, parameters);
  } else {
    return (*layer)(input);
  }
}

template <typename... Ls>
class Layer {
 public:
 public:
  constexpr explicit Layer() : descriptors_(Ls()...) {}

  constexpr explicit Layer(
      const typename std::tuple_element_t<0, std::tuple<Ls...>>& layer)
      : descriptors_({layer}) {}

  constexpr explicit Layer(const std::tuple<Ls...>& layers)
      : descriptors_(layers) {}

  template <typename... Ls2>
  constexpr auto operator|(Layer<Ls2...> layers) const {
    return Layer<Ls..., Ls2...>(std::tuple_cat(descriptors_, layers.layers()));
  }

  constexpr const std::tuple<Ls...>& layers() const { return descriptors_; }

 private:
  std::tuple<Ls...> descriptors_;
};

template <typename... Ls>
class Model;

template <typename... Ls>
constexpr Model<Ls...> MakeModel(const std::tuple<Ls...>& layers) {
  return Model<Ls...>(layers);
}

template <typename Model, size_t LI>
  requires(LI < Model::kLayers)
struct LayerInput {
  using type =
      typename LayerTraits<typename Model::template L<LI - 1>,
                           typename LayerInput<Model, LI - 1>::type>::output_t;
};

template <typename Model>
struct LayerInput<Model, 0> {
  using type = typename Model::input_t;
};

inline float Emancipate(float f) { return f; }
template <typename S>
math::RowMajorMatrix<S::R, S::C> Emancipate(const math::Matrix<S>& m) {
  return math::RowMajorMatrix<S::R, S::C>(m);
}

template <typename... Ls>
class Model {
 public:
  static constexpr auto kLayerIndexes =
      std::make_index_sequence<sizeof...(Ls)>();
  using layers_t = std::tuple<Ls...>;
  template <size_t I>
  using L = std::tuple_element_t<I, layers_t>;
  constexpr static size_t kLayers = sizeof...(Ls);
  using input_t = typename L<0>::input_t;
  template <size_t LI>
  using Traits = LayerTraits<L<LI>, typename LayerInput<Model, LI>::type>;
  using output_t = typename Model::template Traits<sizeof...(Ls) - 1>::output_t;

  template <size_t LI>
  constexpr static size_t LayerParameters = Traits<LI>::parameter_count;

  constexpr explicit Model() = default;

  constexpr explicit Model(std::tuple<Ls...> layers)
      : layers_(std::move(layers)) {}

  template <size_t Ind = 0>
  static constexpr size_t all_parameters_count() {
    constexpr size_t count = LayerParameters<Ind>;
    if constexpr (Ind == sizeof...(Ls) - 1) {
      return count;
    } else {
      return count + all_parameters_count<Ind + 1>();
    }
  }

  auto operator()(const typename Model::input_t& input,
                  const ModelParameters<Model>& parameters) const {
    ContextForInfer<Model, std::remove_cvref_t<decltype(input)>> context;
    auto r = operator()(input, parameters, context);
    return Emancipate(std::move(r));
  }

  auto operator()(const typename Model::input_t& input,
                  const ModelParameters<Model>& parameters,
                  memory::Context<Model, input_t>& context) const {
    return infer_layer<0>(input, parameters, context);
  }

  template <typename L, typename... Ls2>
  constexpr auto operator|(const Layer<L, Ls2...>& layers) const {
    return std::apply(
        [this](const L& layer, const Ls2&... layers) {
          auto new_layer = std::make_tuple(layer.stack(*this));
          if constexpr (sizeof...(Ls2) == 0) {
            return MakeModel(std::tuple_cat(layers_, std::move(new_layer)));
          } else {
            return MakeModel(std::tuple_cat(layers_, std::move(new_layer))) |
                   Layer<Ls2...>(std::make_tuple(layers...));
          }
        },
        layers.layers());
  }

  constexpr const std::tuple<Ls...>& layers() const { return layers_; }

  template <size_t Ind>
  constexpr auto& layer() const {
    return std::get<Ind>(layers_);
  }

 private:
  template <size_t Ind>
  auto infer_layer(const typename L<Ind>::input_t& input,
                   const ModelParameters<Model>& parameters,
                   memory::Context<Model, input_t>& context) const {
    internal::InferenceLayerContext layer_context(
        std::get<Ind>(context.GetLayerArenas())());
    auto intermediate =
        InvokeLayer(&std::get<Ind>(layers_), input,
                    parameters.template layer_parameters<Ind>(), layer_context);
    if constexpr (Ind < sizeof...(Ls) - 1) {
      return infer_layer<Ind + 1>(std::move(intermediate), parameters, context);
    } else {
      return intermediate;
    }
  }

  std::tuple<Ls...> layers_;
};

}  // namespace uchen

#endif  // UCHEN_STATIC_UTILS_H_