#ifndef UCHEN_TENSOR_ARITHMETIC_H
#define UCHEN_TENSOR_ARITHMETIC_H

#include <cstddef>
#include <span>
#include <utility>

#include "uchen/tensor/tensor.h"

namespace uchen::tensor {
namespace internal {

class AddOperation {
 public:
  AddOperation(Layout al, std::span<const float> a, Layout bl,
               std::span<const float> b)
      : al_(std::move(al)), a_(a), bl_(std::move(bl)), b_(b) {}

  void Perform(const Layout& receiver_layout, std::span<float> receiver) const;

 private:
  Layout al_;
  std::span<const float> a_;
  Layout bl_;
  std::span<const float> b_;
};

// Ensures compiler checks that all tensors have same dimensions
template <typename T, size_t D, size_t... Ds>
class TypeSafeWrapper : public TensorOperation<float, D, Ds...> {
 public:
  template <typename... Args>
  explicit TypeSafeWrapper(Args&&... args) : op_(std::forward<Args>(args)...) {}

  void Apply(TensorView<float, D, Ds...>& tensor) const override {
    op_.Perform(tensor.layout(), tensor.data());
  }

 private:
  T op_;
};

}  // namespace internal

template <typename T, size_t D, size_t... Ds>
void Add(TensorView<T, D, Ds...> receiver, const TensorRef<T, D, Ds...>& a,
         const TensorRef<T, D, Ds...>& b) {
  internal::AddOperation adder{a.layout(), a.data(), b.layout(), b.data()};
  adder.Perform(receiver.layout(), receiver.data());
}

template <size_t D, size_t... Ds>
internal::TypeSafeWrapper<internal::AddOperation, D, Ds...> operator+(
    const TensorRef<float, D, Ds...>& a, const TensorRef<float, D, Ds...>& b) {
  return internal::TypeSafeWrapper<internal::AddOperation, D, Ds...>{
      a.layout(), a.data(), b.layout(), b.data()};
}

}  // namespace uchen::tensor

#endif  // UCHEN_TENSOR_ARITHMETIC_H