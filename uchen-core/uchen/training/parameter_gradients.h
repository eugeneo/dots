#ifndef UCHEN_TRAINING_PARAMETER_GRADIENTS_H
#define UCHEN_TRAINING_PARAMETER_GRADIENTS_H

#include <cstddef>
#include <span>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "uchen/memory.h"
#include "uchen/parameters.h"

namespace uchen::training {

template <typename Model>
class ParameterGradients {
 public:
  using value_type = float;

  ParameterGradients() : gradients_(Model::all_parameters_count(), 0.f) {}

  explicit ParameterGradients(const Model* /* inference */)
      : ParameterGradients() {}

  template <size_t L>
  std::span<float, Model::template Traits<L>::parameter_count>
  layer_parameter_gradients() {
    return std::span(gradients_)
        .template subspan<ParametersUpTo<L>(),
                          Model::template Traits<L>::parameter_count>();
  }

  static constexpr size_t size() { return Model::all_parameters_count(); }

  auto begin() const { return gradients_.begin(); }
  auto end() const { return gradients_.end(); }

  float& operator[](size_t i) { return gradients_[i]; }
  const float operator[](size_t i) const { return gradients_[i]; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ParameterGradients& p) {
    absl::Format(&sink, "{ %s }", absl::StrJoin(p.gradients_, ", "));
  }

  ParameterGradients& operator+=(const ParameterGradients& other) {
    for (size_t i = 0; i < gradients_.size(); ++i) {
      gradients_[i] += other.gradients_[i];
    }
    return *this;
  }

  ParameterGradients& operator*=(float f) {
    for (size_t i = 0; i < gradients_.size(); i++) {
      gradients_[i] *= f;
    }
    return *this;
  }

  ParameterGradients& operator/=(float f) {
    for (size_t i = 0; i < gradients_.size(); i++) {
      gradients_[i] /= f;
    }
    return *this;
  }

 private:
  template <size_t L>
  static constexpr size_t ParametersUpTo() {
    if constexpr (L == 0) {
      return 0;
    } else {
      return Model::template Traits<L - 1>::parameter_count +
             ParametersUpTo<L - 1>();
    }
  }

  std::vector<float> gradients_;
};

template <typename Model>
ParameterGradients<Model> operator+(ParameterGradients<Model> a,
                                    const ParameterGradients<Model>& b) {
  return a += b;
}

template <typename Model>
ParameterGradients<Model> operator/(ParameterGradients<Model> a, float b) {
  return a /= b;
}

template <typename Model>
ParameterGradients<Model> operator*(ParameterGradients<Model> a, float b) {
  return a *= b;
}

template <typename Model>
ParameterGradients<Model> operator-(const ModelParameters<Model>& a,
                                    const ModelParameters<Model>& b) {
  ParameterGradients<Model> gradients;
  std::span p1 = a.data();
  std::span p2 = b.data();
  for (size_t i = 0; i < Model::all_parameters_count(); ++i) {
    gradients[i] = p1[i] - p2[i];
  }
  return gradients;
}

template <typename Model>
ModelParameters<Model> operator+(const ModelParameters<Model>& parameters,
                                 const ParameterGradients<Model>& gradients) {
  std::shared_ptr<std::vector<float>> result =
      std::make_shared<std::vector<float>>(*parameters.store());
  for (size_t i = 0; i < Model::all_parameters_count(); i++) {
    (*result)[i] += gradients[i];
  }
  return ModelParameters<Model>(parameters.model(), result);
}

template <typename Model>
ModelParameters<Model> operator-(const ModelParameters<Model>& parameters,
                                 const ParameterGradients<Model>& gradients) {
  auto result = ParametersCopy(parameters);
  std::span span = result->data();
  for (size_t i = 0; i < Model::all_parameters_count(); i++) {
    span[i] -= gradients[i];
  }
  return ModelParameters<Model>(parameters.model(), std::move(result));
}

}  // namespace uchen::training

namespace std {

template <typename M>
std::ostream& operator<<(
    std::ostream& os, const uchen::training::ParameterGradients<M>& gradients) {
  os << absl::StrCat(gradients);
  return os;
}

}  // namespace std

#endif  // UCHEN_TRAINING_PARAMETER_GRADIENTS_H