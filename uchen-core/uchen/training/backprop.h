#ifndef UCHEN_BACKPROP_H_
#define UCHEN_BACKPROP_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include "uchen/layer_traits.h"
#include "uchen/layers.h"
#include "uchen/linear.h"
#include "uchen/math/matrix.h"
#include "uchen/memory.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/softmax.h"
#include "uchen/vector.h"

namespace uchen::training {

template <typename I, typename G>
G ComputeGradients(const InputLayer<I>& layer, const I& input,
                   const G& output_gradients,
                   const Parameters<0>& /* parameters */,
                   std::span<float, 0> /* parameter_gradients */,
                   const void* /* area */) {
  return output_gradients;
}

template <typename I, size_t O>
Vector<float, I::elements> ComputeGradients(
    const Linear<I, O>& layer, const I& input,
    const Vector<float, O>& output_gradients,
    const Parameters<LayerTraits<Linear<I, O>, I>::parameter_count>& parameters,
    std::span<float, LayerTraits<Linear<I, O>, I>::parameter_count>
        parameter_gradients,
    const void* /* area */) {
  // Biases
  std::copy(output_gradients.begin(), output_gradients.end(),
            parameter_gradients.begin());
  math::Matrix og = math::AsColumnMajorView<1, O>(output_gradients.data());
  math::Matrix pars =
      math::AsColumnMajorView<O, I::elements>(std::span(parameters).subspan(O));
  math::Matrix x = math::AsColumnMajorView<1, I::elements>(input);
  auto input_grads =
      memory::DeletableAnything<std::array<float, I::elements>>::NewInstance();

  math::AsColumnMajorView<1, I::elements>(input_grads->get()) = og * pars;
  math::AsColumnMajorView<O, I::elements>(parameter_gradients.subspan(O)) =
      og.transposed() * x;

  std::span s = input_grads->get();
  return Vector<float, I::elements>(s, std::move(input_grads));
}

template <typename V>
float ElementGradient(const Relu& /* relu */, const V& v, float loss_gradient) {
  if (v < 0) {
    return 0;
  } else {
    return loss_gradient;
  }
}

template <typename V>
float ElementGradient(const Sigmoid& /* relu */, const V& v,
                      float loss_gradient) {
  return static_cast<float>(v * (1 - v)) * loss_gradient;
}

template <typename I, typename Op>
Vector<float, I::elements> ComputeGradients(
    const ElementWise<I, Op>& layer, const I& input,
    const Vector<float, I::elements>& output_gradients,
    const Parameters<0>& parameters, std::span<float, 0> parameter_gradients,
    const void* /* area */) {
  auto store = memory::ArrayStore<float, I::elements>::NewInstance();
  std::span span = store->data();
  Op op;
  for (size_t i = 0; i < I::elements; ++i) {
    span[i] = ElementGradient(op, input[i], output_gradients[i]);
  }
  return Vector<float, I::elements>(std::move(store));
}

template <typename C, typename V, size_t S>
Vector<float, S> ComputeGradients(const Categories<C, V, S>& layer,
                                  const Vector<V, S>& input,
                                  const Vector<float, S>& loss_gradient,
                                  const Parameters<0>& /* parameter */,
                                  std::span<float, 0> /* parameter_gradients */,
                                  const void* /* area */) {
  if constexpr (std::is_same_v<V, float>) {
    return input;
  } else {
    auto store = memory::ArrayStore<float, S>::NewInstance();
    std::span data = store->data;
    for (size_t i = 0; i < S; i++) {
      data[i] = input[i];
    }
    return Vector<float, S>(std::move(store));
  }
}

}  // namespace uchen::training

#endif  // UCHEN_BACKPROP_H_