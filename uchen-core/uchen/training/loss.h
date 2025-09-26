#ifndef UCHEN_LOSS_H
#define UCHEN_LOSS_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "absl/log/log.h"

#include "uchen/model.h"
#include "uchen/softmax.h"
#include "uchen/vector.h"

namespace uchen::training {

template <typename C>
struct CrossEntropy {
  using value_type = C;
  using target_t = C;

  template <typename Cat, size_t S>
  Vector<float, S> Gradient(const CategoricalResult<Cat, S>& y,
                            const C& y_hat) const {
    auto sm = y.Softmax();
    auto result =
        memory::ArrayStore<float, S>::NewInstance(sm.begin(), sm.end());
    result->data()[y.IndexOf(y_hat)] -= 1;
    return Vector<float, S>(std::move(result));
  }

  template <size_t S>
  float Loss(const CategoricalResult<C, S>& y, const C& y_hat) const {
    size_t i = -1;
    // SSE instructions for vector multiplication are cool, but brute force here
    // is more efficient.
    for (const auto& [category, weight] : y.MatchDetails()) {
      ++i;
      if (category == y_hat) {
        float v = y.Softmax().GetColumnMajor(i);
        // Weight should never be zero. But is there truly anyone we can trust
        // in this world?
        return -std::log(v == 0 ? std::numeric_limits<float>::min() : v);
      }
    }
    DLOG(FATAL) << "No matching category in softmax result " << y;
    return 0.f;
  }
};

template <typename V, size_t C>
struct SquaredLoss {
  using value_type = Vector<V, C>;

  Vector<float, C> Gradient(const value_type& y,
                            const value_type& y_hat) const {
    std::span yys = y.data();
    std::span yhs = y_hat.data();
    auto store = memory::ArrayStore<float, C>::NewInstance();
    std::span r = store->data();
    for (size_t i = 0; i < C; ++i) {
      r[i] = 2 * (yys[i] - yhs[i]);
    }
    return Vector<float, C>(std::move(store));
  }

  double Loss(const value_type& y, const value_type& y_hat) const {
    std::span yys = y.data();
    std::span yhs = y_hat.data();
    double accum = 0;
    for (size_t i = 0; i < C; ++i) {
      double v = yys[i] - yhs[i];
      accum += v * v;
    }
    return accum / C;
  }
};

template <typename T>
struct DefaultLoss;

template <typename V, size_t C>
struct DefaultLoss<Vector<V, C>> {
  using type = SquaredLoss<V, C>;
};

template <typename C, size_t S>
struct DefaultLoss<CategoricalResult<C, S>> {
  using type = CrossEntropy<C>;
};

}  // namespace uchen::training

#endif  // UCHEN_LOSS_H