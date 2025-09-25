#ifndef UCHEN_LINEAR_H
#define UCHEN_LINEAR_H

#include <stddef.h>

#include <array>
#include <memory>

#include "model.h"
#include "uchen/layer_traits.h"
#include "uchen/math/matrix.h"
#include "uchen/memory.h"
#include "uchen/parameters.h"
#include "uchen/vector.h"

namespace uchen {

template <typename Input, size_t Outputs>
  requires(Outputs > 0)
struct Linear {
  using input_t = Input;

  Vector<typename Input::value_type, Outputs> operator()(
      const input_t& inputs,
      const Parameters<(Input::elements + 1) * Outputs>& parameters,
      memory::LayerContext<std::array<float, Outputs>>* context) const {
    DCHECK_NE(context, nullptr);
    std::array<float, Outputs>* area = context->GetScratchArea();
    math::Matrix x = math::AsColumnMajorView<Input::elements>(inputs.data());
    math::Matrix a = math::AsColumnMajorView<Outputs, Input::elements>(
        parameters.template starting<Outputs>());
    math::Matrix b = math::AsColumnMajorView<Outputs>(
        std::span<const float>(parameters.data(), Outputs));
    math::Matrix y = math::AsColumnMajorView<Outputs>(std::span<float>(*area));
    y = a * x + b;
    return Vector<typename Input::value_type, Outputs>(*area, nullptr);
  }
};

template <typename I, size_t O>
Parameters<(I::elements + 1) * O> ParameterProvider(
    const Linear<I, O>& layer, std::span<const float> data,
    std::shared_ptr<memory::Deletable> ref) {
  return Parameters<(I::elements + 1) * O>(data, std::move(ref));
}

template <size_t Outputs>
struct LinearLayerDesc {
  template <typename Model>
  constexpr auto stack(const Model& /* il */) const {
    return Linear<typename Model::output_t, Outputs>();
  }
};

namespace layers {
template <size_t Outputs>
inline constexpr Layer<LinearLayerDesc<Outputs>> Linear;

}  // namespace layers

template <typename Input, size_t Outputs>
struct LayerTraits<Linear<Input, Outputs>, Input>
    : public LayerTraitFields<Vector<typename Input::value_type, Outputs>,
                              (Input::elements + 1) * Outputs,
                              std::array<float, Outputs>> {};

}  // namespace uchen

#endif  // UCHEN_LINEAR_H