#ifndef UCHEN_SOFTMAX_H
#define UCHEN_SOFTMAX_H

#include <stddef.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "absl/log/check.h"
#include "absl/log/log.h"

#include "uchen/layer_traits.h"
#include "uchen/linear.h"
#include "uchen/math/matrix.h"
#include "uchen/math/softmax.h"
#include "uchen/vector.h"

namespace uchen {

template <typename Category, size_t S>
  requires(S > 0)
class CategoricalResult {
 public:
  using category_t = Category;
  static constexpr size_t elements = S;

  CategoricalResult() = default;

  CategoricalResult(const std::array<Category, S>& categories,
                    Vector<float, S> weights)
      : categories_(categories), weights_(std::move(weights)) {
    best_ = std::numeric_limits<size_t>::max();
    double best_weight = std::numeric_limits<double>::lowest();
    for (size_t i = 0; i < S; ++i) {
      if (best_weight < weights_[i]) {
        best_weight = weights_[i];
        best_ = i;
      }
    }
  }

  bool operator==(const Category& c) const { return best_match() == c; }
  bool operator==(const CategoricalResult& c) const {
    return best_match() == c.best_match();
  }

  std::array<std::pair<Category, double>, S> MatchDetails() const {
    std::array<std::pair<Category, double>, S> details;
    for (size_t i = 0; i < S; ++i) {
      details[i] = {categories_[i], weights_[i]};
    }
    return details;
  }

  Category best_match() const {
    DCHECK_LT(best_, S);
    return categories_[best_];
  }

  Vector<float, S> raw_weights() const { return weights_; }
  std::array<Category, S> categories() const { return categories_; }

  math::ColumnMajorMatrix<S, 1> Softmax() const {
    // Now the range is [0, 1], double is overkill. Typecast is fine, there will
    // be a limited number of categories.
    return math::Softmax(
        math::AsColumnMajorView<S, 1>(std::span<const float>(weights_.data())));
  }

  template <std::equality_comparable_with<Category> C>
  size_t IndexOf(const C& category) const {
    for (size_t i = 0; i < categories_.size(); i++) {
      if (categories_[i] == category) {
        return i;
      }
    }
    DLOG(FATAL) << "Category " << category << " not found";
    return -1;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const CategoricalResult& category) {
    if constexpr (std::is_arithmetic_v<Category>) {
      absl::Format(&sink, "C(%d){", category.best_match());
    } else {
      absl::Format(&sink, "C(%v){", category.best_match());
    }
    math::Matrix softmax = category.Softmax();
    auto it = softmax.begin_across();
    for (size_t i = 0; i < S; ++i) {
      if constexpr (std::is_arithmetic_v<Category>) {
        absl::Format(&sink, "%d: %2.2f", category.categories_[i], *it++);
      } else {
        absl::Format(&sink, "%v: %2.2f", category.categories_[i], *it++);
      }
      if (i < S - 1) {
        absl::Format(&sink, ", ");
      }
    }
    absl::Format(&sink, " }");
  }

  friend void PrintTo(const CategoricalResult& result, std::ostream* os) {
    *os << absl::StrFormat("%v", result);
  }

 private:
  std::array<Category, S> categories_;
  Vector<float, S> weights_;
  size_t best_;
};

template <typename Category, size_t S>
CategoricalResult<Category, S> Emancipate(
    const CategoricalResult<Category, S>& result) {
  return CategoricalResult<Category, S>(result.categories(),
                                        result.raw_weights().Emancipate());
}

template <typename Category, typename V, size_t S>
  requires(S > 0)
class Categories {
 public:
  using input_t = Vector<V, S>;
  using output_t = CategoricalResult<Category, S>;
  constexpr static size_t parameter_count = 0;

  constexpr explicit Categories(const std::array<Category, S>& categories)
      : categories_(categories) {
    DCHECK_EQ(categories_.size(), S);
  }

  CategoricalResult<Category, S> operator()(const input_t& inputs) const {
    DCHECK_EQ(inputs.size(), S);
    auto store = memory::ArrayStore<float, S>::NewInstance();
    std::copy(inputs.begin(), inputs.end(), store->data().begin());
    return CategoricalResult<Category, S>(categories_,
                                          Vector<float, S>(std::move(store)));
  }

  constexpr std::span<const Category, S> categories() const {
    return categories_;
  }

 private:
  std::array<Category, S> categories_;
};

template <typename Cat, typename ET, size_t C>
struct LayerTraits<Categories<Cat, ET, C>, Vector<ET, C>>
    : public LayerTraitFields<CategoricalResult<Cat, C>> {};

template <typename Category, size_t S>
class SoftmaxLayerDesc {
 public:
  constexpr SoftmaxLayerDesc(const std::array<Category, S>& categories)
      : categories_(categories) {}

  template <typename Layer>
  constexpr Categories<Category, typename Layer::output_t::value_type, S> stack(
      const Layer& /* layer */) const {
    return Categories<Category, typename Layer::output_t::value_type, S>(
        categories_);
  }

 private:
  std::array<Category, S> categories_;
};

namespace layers {

template <typename Category, size_t S>
constexpr auto Categories(const std::array<Category, S>& categories) {
  return layers::Linear<S> |
         Layer<SoftmaxLayerDesc<Category, S>>(SoftmaxLayerDesc(categories));
}

}  // namespace layers
}  // namespace uchen

#endif  // UCHEN_SOFTMAX_H