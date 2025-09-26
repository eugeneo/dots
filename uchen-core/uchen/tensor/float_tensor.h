#ifndef UCHEN_TENSOR_FLOAT_TENSOR_H
#define UCHEN_TENSOR_FLOAT_TENSOR_H

#include <array>
#include <concepts>
#include <cstddef>
#include <span>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"  // IWYU pragma: keep
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

namespace uchen::core {

class BasicTensor;

namespace details {

constexpr bool SameExceptFirst(std::span<const size_t> a,
                               std::span<const size_t> b) {
  if (a.size() != b.size() + 1) return false;
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i] != b[i - 1]) return false;
  }
  return true;
}

class DimSliceTranslator {
 public:
  constexpr DimSliceTranslator(size_t dim, size_t start, size_t size)
      : dim_(dim), start_(start), size_(size) {}

  size_t dim(size_t dim, const BasicTensor& input) const;
  size_t Translate(size_t index, const BasicTensor& input) const;

 private:
  size_t dim_;
  size_t start_;
  size_t size_;
};

class TransposeTranslator {
 public:
  size_t dim(size_t dim, const BasicTensor& input) const;
  size_t Translate(size_t index, const BasicTensor& input) const;
};

}  // namespace details

template <typename T>
concept TensorLike = requires(T) {
  { T::kDims } -> std::convertible_to<std::span<const size_t>>;
  (T::kDims.size() > 0);
};

template <typename T, size_t... Dims>
concept BatchTensorWithDims =
    TensorLike<T> &&
    details::SameExceptFirst(T::kDims,
                             std::array<size_t, sizeof...(Dims)>{Dims...});

class TensorProjection;

class BasicTensor {
 public:
  virtual ~BasicTensor() = default;
  virtual size_t dim(size_t dim) const = 0;
  virtual size_t rank() const = 0;
  virtual float get(size_t index) const = 0;

  template <std::convertible_to<size_t>... Indices>
  float operator()(Indices... indices) const {
    DCHECK_EQ(sizeof...(Indices), rank());
    std::array<size_t, sizeof...(Indices)> indices_array = {
        static_cast<size_t>(indices)...};
    size_t index = indices_array.front();
    for (size_t i = 1; i < indices_array.size(); ++i) {
      index = index * dim(i) + indices_array[i];
    }
    return get(index);
  }

  friend void AbslStringify(auto& sink, const BasicTensor& tensor) {
    absl::InlinedVector<size_t, 256> dims;
    for (size_t i = 0; i < tensor.rank(); ++i) {
      dims.push_back(tensor.dim(i));
    }
    sink.Append(absl::Substitute("[$0]", absl::StrJoin(dims, "x")));
  }

  std::string dims_string() const {
    std::string result;
    for (size_t i = 0; i < rank(); ++i) {
      if (i == 0) {
        absl::StrAppend(&result, dim(i));
      } else {
        absl::StrAppend(&result, "x", dim(i));
      }
    }
    return result;
  }

  size_t elements() const {
    size_t elements = 1;
    for (size_t i = 0; i < rank(); ++i) {
      elements *= dim(i);
    }
    return elements;
  }

 protected:
  bool same_dims(const BasicTensor& other) const {
    if (rank() != other.rank()) return false;
    for (size_t i = 0; i < rank(); ++i) {
      if (dim(i) != other.dim(i)) return false;
    }
    return true;
  }
};

class TensorProjection : public virtual BasicTensor {
 public:
  using Translators =
      std::variant<details::DimSliceTranslator, details::TransposeTranslator>;

  static TensorProjection dim_slice(const BasicTensor& tensor
                                        ABSL_ATTRIBUTE_LIFETIME_BOUND,
                                    size_t dim, size_t start, size_t size) {
    return {tensor, details::DimSliceTranslator{dim, start, size}};
  }

  TensorProjection(const BasicTensor& input, Translators translator)
      : input_(input), translator_(std::move(translator)) {}

  size_t rank() const final { return input_.rank(); }
  size_t dim(size_t dim) const final {
    return std::visit(
        [&](const auto& translator) { return translator.dim(dim, input_); },
        translator_);
  }

  float get(size_t index) const final {
    size_t translated = std::visit(
        [&](const auto& translator) {
          return translator.Translate(index, input_);
        },
        translator_);
    return input_.get(translated);
  }

 protected:
  const BasicTensor& input_;
  const Translators translator_;
};

class AssignableTensor : public virtual BasicTensor {
 public:
  class Assignable {
   public:
    virtual ~Assignable() = default;
  };
  virtual ~AssignableTensor() = default;
  virtual AssignableTensor& operator=(const Assignable&) = 0;
  virtual void set(size_t index, float value) = 0;

 protected:
  static void assign(AssignableTensor& self, const BasicTensor& other) {
    DCHECK(self.same_dims(other))
        << "Incompatible tensor dimensions: " << self.dims_string() << " vs "
        << other.dims_string();
    for (size_t i = 0; i < self.elements(); ++i) {
      self.set(i, other.get(i));
    }
  }

  static void assign(AssignableTensor& self, const Assignable& other) {}
};

class MultiplicationResult final : public AssignableTensor::Assignable {
 public:
  MultiplicationResult(const BasicTensor& a, const BasicTensor& b) {}
  // : a_(a), b_(b) {}

 private:
  // const BasicTensor& a_;
  // const BasicTensor& b_;
};

class AssignableProjection final : public TensorProjection,
                                   public AssignableTensor {
 public:
  static AssignableProjection dim_slice(AssignableTensor& tensor
                                            ABSL_ATTRIBUTE_LIFETIME_BOUND,
                                        size_t dim, size_t start, size_t size) {
    return {tensor, details::DimSliceTranslator{dim, start, size}};
  }

  AssignableProjection(AssignableTensor& input,
                       TensorProjection::Translators translator)
      : TensorProjection(input, std::move(translator)), data_(input) {}

  AssignableTensor& operator=(const Assignable&) override { return *this; }
  AssignableProjection& operator=(const BasicTensor& other) {
    AssignableTensor::assign(*this, other);
    return *this;
  }

  void set(size_t index, float value) override { data_.set(index, value); }

 private:
  AssignableTensor& data_;
};

inline MultiplicationResult operator*(const BasicTensor& a,
                                      const BasicTensor& b) {
  return {a, b};
}

template <size_t... Dims>
  requires(sizeof...(Dims) > 0)
class FloatTensor final : public AssignableTensor {
 public:
  static constexpr size_t kDims[sizeof...(Dims)] = {Dims...};

  FloatTensor() = default;
  FloatTensor(FloatTensor&&) = default;
  FloatTensor(const Assignable& assignable) { assign(*this, assignable); }

  size_t dim(size_t dim) const override { return kDims[dim]; }
  size_t rank() const override { return sizeof...(Dims); }
  FloatTensor& operator=(const Assignable& other) override {
    assign(*this, other);
    return *this;
  }

  template <std::integral... T>
    requires(sizeof...(T) == sizeof...(Dims))
  float& operator()(T... is) {
    std::array indices = {is...};
    size_t index = indices.front();
    for (size_t i = 1; i < sizeof...(Dims); ++i) {
      index = index * kDims[i] + indices[i];
    }
    return data_[index];
  }

  float& operator()(std::span<const size_t, sizeof...(Dims)> indices) {
    size_t index = indices.front();
    for (size_t i = 1; i < sizeof...(Dims); ++i) {
      index = index * kDims[i] + indices[i];
    }
    return data_[index];
  }

  std::span<const float, (Dims * ...)> data() const { return data_; }

  float get(size_t index) const override { return data_[index]; }
  void set(size_t index, float value) override { data_[index] = value; }

 private:
  std::array<float, (Dims * ...)> data_;
};

}  // namespace uchen::core

#endif  // UCHEN_TENSOR_FLOAT_TENSOR_H
