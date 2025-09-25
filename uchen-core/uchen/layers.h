#ifndef UCHEN_V2_H_
#define UCHEN_V2_H_

#include <stddef.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>

#include "uchen/layer_traits.h"
#include "uchen/memory.h"
#include "uchen/model.h"
#include "uchen/vector.h"

namespace uchen {

// This should not exist!
template <typename V>
struct InputLayer {
  using input_t = V;
  using output_t = V;

  V operator()(const V& inputs) const { return inputs; }
};

template <typename V>
struct LayerTraits<InputLayer<V>, V>
    : public LayerTraitFields<V, 0, details::Empty, true> {};

template <typename V, size_t HiddenState>
using WithHiddenState = std::pair<V, Vector<float, HiddenState>>;

struct Relu {
  auto operator()(auto x) { return x > 0 ? x : 0; }
};

struct Sigmoid {
  auto operator()(auto x) { return 1 / (1 + std::exp(-x)); }
};

template <typename V, typename Op>
struct ElementWise {
  using input_t = V;
  using output_t = input_t;
  constexpr static size_t parameter_count = 0;

  output_t operator()(
      const input_t& inputs,
      memory::LayerContext<
          memory::ArrayStore<typename V::value_type, V::elements>>* context =
          nullptr) const {
    std::span<typename V::value_type> data;
    std::unique_ptr<memory::ArrayStore<typename V::value_type, V::elements>>
        store;
    if (context != nullptr) {
      data = context->GetScratchArea()->data();
    } else {
      store = memory::ArrayStore<typename V::value_type,
                                 V::elements>::NewInstance();
      data = store->data();
    }
    Op op;
    for (size_t i = 0; i < V::elements; ++i) {
      data[i] = op(inputs[i]);
    }
    return output_t(data.template first<V::elements>(), std::move(store));
  }
};

template <typename V, typename Op>
struct LayerTraits<ElementWise<V, Op>, V> : public VectorOutputLayer<V> {};

namespace layers {

namespace descriptors {

template <typename Op>
struct ElementWiseDesc {
  template <typename Model>
  constexpr auto stack(const Model& /* il */) const {
    return ElementWise<typename Model::output_t, Op>();
  }
};

}  // namespace descriptors

template <typename V>
inline constexpr Model<InputLayer<V>> Input;

inline constexpr Layer<descriptors::ElementWiseDesc<Relu>> Relu;
inline constexpr Layer<descriptors::ElementWiseDesc<Sigmoid>> Sigmoid;

}  // namespace layers
}  // namespace uchen

#endif  // UCHEN_V2_H_
