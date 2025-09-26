#ifndef UCHEN_TENSOR_SPECIAL_TENSORS_H
#define UCHEN_TENSOR_SPECIAL_TENSORS_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/log/check.h"
#include "absl/log/log.h"  // IWYU pragma: keep

#include "uchen/tensor/float_tensor.h"

namespace uchen::core {

template <size_t Dim, size_t... Dims>
class OneHotTensor;

namespace internal {}  // namespace internal

/*
 * Space-efficient One Hot tensor representation.
 */
template <size_t Dim, size_t... Dims>
class OneHotTensor {
 public:
  static constexpr std::array kDims = {Dim, Dims...};

  class OneHotProjection {
   public:
    static constexpr std::array kDims = {Dims...};

    explicit OneHotProjection(std::span<uint32_t> data) : data_(data) {
      DCHECK_EQ(data_.size(), (Dims * ...) / kDims.back());
    }

    OneHotProjection(const OneHotProjection& other) = delete;
    OneHotProjection& operator=(const OneHotProjection& other) = delete;
    OneHotProjection(OneHotProjection&& other) = delete;
    OneHotProjection& operator=(OneHotProjection&& other) = delete;

    OneHotProjection& operator=(const OneHotTensor<Dims...>& other) {
      std::copy(other.data().begin(), other.data().end(), data_.begin());
      return *this;
    }

   private:
    std::span<uint32_t> data_;
  };

  template <size_t Ds = sizeof...(Dims) + 1>
    requires(Ds > 1)
  OneHotProjection operator[](size_t index) {
    return OneHotProjection(std::span<uint32_t>(data_).subspan(
        index * (Dims * ...) / kDims.back(), (Dims * ...) / kDims.back()));
  }

  std::span<uint32_t> data() { return data_; }
  std::span<const uint32_t> data() const { return data_; }

 private:
  std::array<uint32_t, Dim*(Dims*...) / kDims.back()> data_;
};

// Tensor that does not own the data, instead it provides a view into existing
// data.
template <size_t Dim, size_t... Dims>
class RowProjectionsTensor final : public BasicTensor {
 public:
  static constexpr std::array kDims = {Dim, Dims...};

  constexpr size_t dim(size_t dim) const override {
    DCHECK_LT(dim, kDims.size());
    return kDims[dim];
  }
  size_t rank() const override { return sizeof...(Dims) + 1; }

  std::span<std::span<const float>> flat_data() { return data_; }
  std::span<const std::span<const float>> flat_data() const { return data_; }

  float get(size_t indices) const override {
    DCHECK_LT(indices, Dim * ... * Dims);
    size_t span = indices / kDims.back();
    if (data_[span].size() == 0) {
      return 0.0f;
    }
    return data_[span][indices % kDims.back()];
  }

 private:
  std::array<std::span<const float>, Dim*(Dims*...) / kDims.back()> data_;
};

}  // namespace uchen::core

#endif  // UCHEN_TENSOR_SPECIAL_TENSORS_H