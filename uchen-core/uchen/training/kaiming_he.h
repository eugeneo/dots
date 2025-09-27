#ifndef UCHEN_TRAINING_HE_H
#define UCHEN_TRAINING_HE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <utility>

#include "uchen/linear.h"
#include "uchen/memory.h"
#include "uchen/model.h"
#include "uchen/parameters.h"
#include "uchen/rnn.h"

namespace uchen::training {

inline absl::AnyInvocable<float()> UniformDistribution() {
  std::random_device rd;   // a seed source for the random number engine
  std::mt19937 gen(rd());  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distrib(0, 1);
  return [gen, distrib]() mutable { return distrib(gen); };
}

template <typename M>
ModelParameters<M> KaimingHeInitializedParameters(
    const M* model,
    absl::AnyInvocable<float()> distribution = UniformDistribution());

namespace internal {

// Parameter store initialized according to Kaiming He. This is a decent
// starting point for training networks.
template <typename M>
class KaimingHeParameterStore final : public Store {
 public:
  using span_and_handle =
      std::pair<std::span<const float>, std::shared_ptr<memory::Deletable>>;

  explicit KaimingHeParameterStore(const M* model,
                                   absl::AnyInvocable<float()> distribution)
      : model_(model), distribution_(std::move(distribution)) {}
  span_and_handle GetLayerParameters(size_t layer) override {
    if (!processed_[layer].has_value()) {
      processed_[layer] = layer_initializers_[layer](this);
    }
    return *processed_[layer];
  }

 private:
  using fn = absl::AnyInvocable<span_and_handle(KaimingHeParameterStore*)>;

  template <typename V, size_t I, size_t O>
  span_and_handle GetLayerParameters(const Linear<Vector<V, I>, O>& layer) {
    std::shared_ptr handle =
        memory::ArrayStore<float, (I + 1) * O>::NewInstance();
    std::span data = handle->data();
    std::fill(data.begin(), data.begin() + O, 0);
    const float stddev = std::sqrt(2.0f / I);
    for (float& point : data.subspan(O)) {
      point = distribution_() * stddev;
    }
    return {handle->data(), handle};
  }

  template <typename I, typename M1, size_t Hs, typename Nl>
  span_and_handle GetLayerParameters(const RnnLayer<I, M1, Hs, Nl>& layer) {
    ModelParameters mm_parameters = KaimingHeInitializedParameters(
        layer.model(), [this]() { return this->distribution_(); });
    ModelParameters hh_parameters = KaimingHeInitializedParameters(
        layer.hh_model(), [this]() { return this->distribution_(); });
    std::shared_ptr store =
        memory::ArrayStore<float,
                           M1::all_parameters_count() +
                               decltype(hh_parameters)::P>::NewInstance();
    auto last = std::copy(mm_parameters.begin(), mm_parameters.end(),
                          store->data().begin());
    std::copy(hh_parameters.begin(), hh_parameters.end(), last);
    return {store->data(), store};
  }

  template <typename L>
  span_and_handle GetLayerParameters(const L& layer) {
    constexpr size_t c = LayerTraits<L, typename L::input_t>::parameter_count;
    if constexpr (c == 0) {
      return {};
    }
    std::shared_ptr handle = memory::ArrayStore<float, c>::NewInstance();
    std::span data = handle->data();
    for (float& point : data) {
      // Sqrt not yet constexpr in C++20
      if constexpr (requires { L::kKaimingHeScaleSquared; }) {
        point = std::sqrt(static_cast<float>(L::kKaimingHeScaleSquared)) *
                distribution_();
      } else {
        point = 4 * distribution_() / c;
      }
    }
    return {handle->data(), handle};
  }

  template <size_t... Is>
  static constexpr std::array<fn, M::kLayers> BuildLayerInitializers(
      std::index_sequence<Is...> /* unused */) {
    return std::array<fn, M::kLayers>{(fn([](KaimingHeParameterStore* store) {
      return store->GetLayerParameters(store->model_->template layer<Is>());
    }))...};
  }

  std::array<fn, M::kLayers> layer_initializers_ =
      BuildLayerInitializers(M::kLayerIndexes);
  const M* model_;
  absl::AnyInvocable<float()> distribution_;
  std::array<std::optional<span_and_handle>, M::kLayers> processed_;
};

}  // namespace internal

// Linear layer biases initted to 0, weights are U(4 / (Inputs + Outputs))
template <typename M>
ModelParameters<M> KaimingHeInitializedParameters(
    const M* model, absl::AnyInvocable<float()> distribution) {
  return ModelParameters<M>(
      model, std::make_shared<internal::KaimingHeParameterStore<M>>(
                 model, std::move(distribution)));
}

}  // namespace uchen::training

#endif  // UCHEN_TRAINING_HE_H