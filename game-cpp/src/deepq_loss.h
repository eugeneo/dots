#ifndef SRD_DEEPQ_LOSS_H
#define SRD_DEEPQ_LOSS_H

#include <stddef.h>

#include "absl/log/check.h"

#include "uchen/vector.h"

namespace uchen::learning {

struct DeepQExpectation {
  size_t action;
  float bellman_target;
};

struct DeepQLoss {
  using value_type = DeepQExpectation;

  template <size_t C>
  Vector<float, C> Gradient(const Vector<float, C>& y,
                            const DeepQExpectation& y_hat) const {
    CHECK_LT(y_hat.action, C);
    auto result = memory::ArrayStore<float, C>::NewInstance(0);
    result->data()[y_hat.action] = y[y_hat.action] - y_hat.bellman_target;
    return Vector<float, C>(std::move(result));
  }

  template <size_t C>
  float Loss(const Vector<float, C>& y, const DeepQExpectation& y_hat) const {
    CHECK_LT(y_hat.action, C);
    float loss = y[y_hat.action] - y_hat.bellman_target;
    return loss * loss;
  }
};

}  // namespace uchen::learning

#endif  // SRD_DEEPQ_LOSS_H