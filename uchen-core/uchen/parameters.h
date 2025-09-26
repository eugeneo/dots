#ifndef UCHEN_CORE_H_
#define UCHEN_CORE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"

#include "uchen/memory.h"

namespace uchen {

class Store : public memory::Deletable {
 public:
  // Returns flat parameter store for a single layer.
  // First return value is a span that points to the store that is big enough
  // Store should exist at least as long as second reference is held
  virtual std::pair<std::span<const float>, std::shared_ptr<memory::Deletable>>
  GetLayerParameters(size_t layer) = 0;
};

template <typename Model>
class ModelParameters;

namespace internal {

// Exposes model parameters organization to runtime code
template <typename Model>
class LayerIndexes {
 public:
  static std::pair<size_t, size_t> start_end(size_t layer) {
    if (layer >= Model::kLayers) {
      return {Model::all_parameters_count(), Model::all_parameters_count()};
    }
    return {indexes_[layer], indexes_[layer + 1]};
  }

  static size_t layer_for_index(size_t index) {
    if (index >= indexes_.back()) {
      return Model::kLayers;
    }
    size_t start = 0;
    size_t end = indexes_.size() - 1;
    while (end - start > 1) {
      size_t mid = std::midpoint(start, end);
      if (index >= indexes_[mid]) {
        start = mid;
      } else if (index < indexes_[mid]) {
        end = mid;
      }
    }
    return start;
  }

 private:
  template <size_t... Is>
  static constexpr std::array<size_t, Model::kLayers + 1> GetLayerStartIndexes(
      std::index_sequence<Is...> /* unused */) {
    std::array<size_t, Model::kLayers + 1> arr = {
        0, Model::template Traits<Is>::parameter_count...};
    for (size_t i = 0; i < arr.size() - 1; ++i) {
      arr[i + 1] += arr[i];
    }
    return arr;
  }

  static constexpr std::array<size_t, Model::kLayers + 1> indexes_ =
      GetLayerStartIndexes(Model::kLayerIndexes);
};

template <typename Model>
class FlatStore final : public Store,
                        public std::enable_shared_from_this<FlatStore<Model>> {
 public:
  FlatStore() = default;
  FlatStore(const float init) { std::fill(data_.begin(), data_.end(), init); }
  FlatStore(std::initializer_list<const float> init) {
    data_.fill(0.f);
    for (size_t i = 0; i < std::min(init.size(), Model::all_parameters_count());
         ++i) {
      data_[i] = init.begin()[i];
    }
  }
  FlatStore(std::span<const float> init)
      : FlatStore(init.begin(),
                  std::min(init.end(),
                           init.begin() + Model::all_parameters_count())) {}
  FlatStore(std::forward_iterator auto begin, std::forward_iterator auto end) {
    data_.fill(0.f);
    std::copy(begin, end, data_.data());
  }

  virtual std::pair<std::span<const float>, std::shared_ptr<memory::Deletable>>
  GetLayerParameters(size_t layer) override {
    auto [start, end] = LayerIndexes<Model>::start_end(layer);
    return {std::span<const float>(data_).subspan(start, end - start),
            this->shared_from_this()};
  }

  std::span<float> data() { return data_; }

 private:
  std::array<float, Model::all_parameters_count()> data_;
};

template <typename Model>
class ModelParametersIterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = float;

  // Needed to be an iterator for reason unknown
  ModelParametersIterator() : parameters_(nullptr), index_(0) {}

  ModelParametersIterator(const ModelParameters<Model>* parameters,
                          size_t index)
      : parameters_(parameters), index_(index) {}

  ModelParametersIterator operator++(int /* unused */) {
    ModelParametersIterator r = *this;
    index_ += 1;
    return r;
  }

  ModelParametersIterator& operator++() {
    index_ += 1;
    return *this;
  }

  ModelParametersIterator operator--(int /* unused */) {
    ModelParametersIterator r = *this;
    index_ -= 1;
    return r;
  }

  ModelParametersIterator& operator--() {
    index_ -= 1;
    return *this;
  }

  ModelParametersIterator& operator+=(difference_type offset) {
    index_ += offset;
    return *this;
  }

  ModelParametersIterator& operator-=(difference_type offset) {
    index_ -= offset;
    return *this;
  }

  std::ptrdiff_t operator-(const ModelParametersIterator& other) const {
    return index_ - other.index_;
  }

  value_type operator[](difference_type offset) const {
    return *(*this + offset);
  }

  value_type operator*() const {
    size_t layer = internal::LayerIndexes<Model>::layer_for_index(index_);
    DCHECK(layer < Model::kLayers);
    size_t index =
        index_ - internal::LayerIndexes<Model>::start_end(layer).first;
    auto [span, handle] = parameters_->parameters()->GetLayerParameters(layer);
    return span[index];
  }

  int operator<=>(const ModelParametersIterator& other) const {
    if (parameters_ < other.parameters_) {
      return -1;
    } else if (parameters_ > other.parameters_) {
      return 1;
    } else if (index_ < other.index_) {
      return -1;
    } else if (index_ > other.index_) {
      return 1;
    } else {
      return 0;
    }
  }

  bool operator==(const ModelParametersIterator& other) const {
    return (*this <=> other) == 0;
  }

  bool operator!=(const ModelParametersIterator& other) const {
    return (*this <=> other) != 0;
  }

 private:
  const ModelParameters<Model>* parameters_;
  size_t index_;
};

template <typename Model>
ModelParametersIterator<Model> operator+(ModelParametersIterator<Model> it,
                                         std::ptrdiff_t offset) {
  return it += offset;
}

template <typename Model>
ModelParametersIterator<Model> operator+(std::ptrdiff_t offset,
                                         ModelParametersIterator<Model> it) {
  return it += offset;
}

template <typename Model>
ModelParametersIterator<Model> operator-(ModelParametersIterator<Model> it,
                                         std::ptrdiff_t offset) {
  return it -= offset;
}

}  // namespace internal

template <typename Model, typename... Args>
std::shared_ptr<internal::FlatStore<Model>> NewFlatStore(
    const Model* model, std::initializer_list<const float> args) {
  return std::make_shared<internal::FlatStore<Model>>(std::move(args));
}

template <typename Model, typename... Args>
std::shared_ptr<internal::FlatStore<Model>> NewFlatStore(const Model* model,
                                                         Args... args) {
  return std::make_shared<internal::FlatStore<Model>>(
      std::forward<Args>(args)...);
}

template <size_t Len>
  requires(Len >= 0)
class Parameters {
 public:
  using value_type = float;
  static constexpr size_t Size = Len;

  Parameters(std::nullptr_t /* null */,
             std::span<const float, 0> /* nothing */) {}
  Parameters(std::span<const float> data,
             std::shared_ptr<const memory::Deletable> data_ref = nullptr)
      : data_(data.template first<Size>()), ref_(std::move(data_ref)) {
    DCHECK_GE(data.size(), Size);
  }
  Parameters(const Parameters& other) : data_(other.data_), ref_(other.ref_) {}

  constexpr size_t size() const { return Len; }

  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }

  float operator[](size_t i) const {
    DCHECK_LT(i, data_.size());
    return data_[i];
  }

  float sum() const {
    float r = 0;
    for (float x : data_) {
      r += x;
    }
    return r;
  }

  template <size_t O, size_t L = Len - O>
    requires(O >= 0 && O + L <= Len)
  Parameters<L> starting() const {
    return Parameters<L>(data_.subspan(O).template first<L>(), ref_);
  }

  const float* data() const { return data_.data(); };

  operator std::span<const float, Len>() const { return data_; }

 private:
  std::span<const float, Len> data_;
  // Holds a ref to the data store so the span above does not outlive it
  std::shared_ptr<const memory::Deletable> ref_;
};

template <>
class Parameters<0> {};

Parameters<0> ParameterProvider(const auto& layer,
                                std::span<const float> /* data */,
                                std::shared_ptr<memory::Deletable> /* ref */) {
  return {};
}

template <typename Model>
class ModelParameters {
 private:
  template <size_t... L>
  static auto ParameterStores(const Model* model,
                              const std::index_sequence<L...>& /* ignore */) {
    auto tup =
        std::make_tuple(ParameterProvider(model->template layer<L>())...);
    return std::make_shared<decltype(tup)>(std::move(tup));
  }

 public:
  using value_type = float;
  static constexpr size_t P = Model::all_parameters_count();

  ModelParameters(const Model* model, std::initializer_list<const float> init)
      : ModelParameters(model, NewFlatStore(model, init)) {}

  explicit ModelParameters(const Model* model, float init = 0)
      : ModelParameters(model, NewFlatStore(model, init)) {}

  ModelParameters(const Model* model, std::span<const float> parameters)
      : ModelParameters(model, NewFlatStore(model, parameters)) {}

  ModelParameters(const Model* model, std::shared_ptr<Store> parameters)
      : model_(model), parameters_(std::move(parameters)) {
    DCHECK_NE(parameters_, nullptr);
    DCHECK_NE(model_, nullptr);
  }

  internal::ModelParametersIterator<Model> begin() const {
    return internal::ModelParametersIterator<Model>(this, 0);
  }

  internal::ModelParametersIterator<Model> end() const {
    return internal::ModelParametersIterator<Model>(
        this, Model::all_parameters_count());
  }

  template <size_t LI, typename L = typename Model::template L<LI>>
    requires(LI < Model::kLayers)
  Parameters<Model::template LayerParameters<LI>> layer_parameters() const {
    auto [span, store] = parameters_->GetLayerParameters(LI);
    return ParameterProvider(model_->template layer<LI>(), span, store);
  }

  constexpr size_t size() const { return P; }

  const Model* model() const { return model_; }

  auto parameters() const { return parameters_; }

 private:
  const Model* model_ = nullptr;
  std::shared_ptr<Store> parameters_;
};

template <typename Model>
ModelParameters<Model>::ModelParametersIterator operator+(
    std::ptrdiff_t offset,
    typename ModelParameters<Model>::ModelParametersIterator it) {
  return it + offset;
}

template <typename M>
ModelParameters<M> RandomParameters(const M* m, float min, float max,
                                    int seed) {
  int concurrency = std::max(std::thread::hardware_concurrency(), 4u);
  auto store = NewFlatStore(m);
  std::span span = store->data();
  std::vector<std::thread> threads;
  for (ptrdiff_t i = 0; i < concurrency; ++i) {
    size_t start = span.size() / concurrency * i;
    threads.emplace_back(
        [](std::span<float> data, int seed, auto concurrency, auto min,
           auto max) {
          std::default_random_engine re(seed);
          std::uniform_real_distribution<float> dist(min, max);
          for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dist(re);
          }
        },
        span.subspan(start,
                     std::min(span.size() - start, span.size() / concurrency)),
        seed, concurrency, min, max);
  }
  for (auto& t : threads) {
    t.join();
  }
  return ModelParameters(m, std::move(store));
}

template <typename Model>
std::shared_ptr<internal::FlatStore<Model>> ParametersCopy(
    const ModelParameters<Model>& parameters) {
  return NewFlatStore(parameters.model(), parameters.begin(), parameters.end());
}

}  // namespace uchen

namespace std {

template <size_t Len>
std::ostream& operator<<(std::ostream& os,
                         const uchen::Parameters<Len>& parameters) {
  os << "Parameters<" << Len << ">[";
  for (size_t i = 0; i < Len; ++i) {
    os << parameters[i];
    if (i < Len - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

template <typename M>
std::ostream& operator<<(std::ostream& os,
                         const uchen::ModelParameters<M>& parameters) {
  os << "ModelParameters<" << M::all_parameters_count() << ">[";
  bool first = true;
  for (const auto& p : parameters) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << p;
  }
  os << "]";
  return os;
}

}  // namespace std

#endif  // UCHEN_CORE_H_