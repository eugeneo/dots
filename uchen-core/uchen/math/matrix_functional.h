#ifndef UCHEN_MATH_MATRIX_FUNCTIONAL_H
#define UCHEN_MATH_MATRIX_FUNCTIONAL_H

#include <cstddef>
#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"  // IWYU pragma: keep

#include "uchen/math/matrix_core.h"

namespace uchen::math {

namespace details {

template <uint32_t R, uint32_t C>
class TwoArgFn : public MatrixWithSize<R, C> {
 public:
  explicit TwoArgFn(absl::AnyInvocable<float(uint32_t, uint32_t) const> fn)
      : fn_(std::move(fn)) {}

  constexpr float GetRowMajor(size_t i) const final {
    return fn_(static_cast<uint32_t>(i / C), static_cast<uint32_t>(i % C));
  }

  constexpr float GetColumnMajor(size_t i) const {
    return fn_(static_cast<uint32_t>(i % R), static_cast<uint32_t>(i / R));
  }

  ReadonlyRowIterator begin() const { return ReadonlyRowIterator(this, 0); }

  ReadonlyRowIterator end() const {
    return ReadonlyRowIterator(this, this->size());
  }

  ReadonlyColumnIterator begin_across() const {
    return ReadonlyColumnIterator(this, 0);
  }

  ReadonlyColumnIterator end_across() const {
    return ReadonlyColumnIterator(this, this->size());
  }

 private:
  absl::AnyInvocable<float(uint32_t, uint32_t) const> fn_;
};

template <uint32_t R, uint32_t C>
class SingleArgFn : public MatrixWithSize<R, C> {
 public:
  explicit SingleArgFn(absl::AnyInvocable<float(size_t) const> fn)
      : fn_(std::move(fn)) {}

  constexpr float GetRowMajor(size_t i) const final { return fn_(i); }

  constexpr float GetColumnMajor(size_t i) const {
    return fn_(details::ConvertIndexDirection(i, R, C));
  }

  ReadonlyRowIterator begin() const { return ReadonlyRowIterator(this, 0); }
  ReadonlyRowIterator end() const {
    return ReadonlyRowIterator(this, this->size());
  }

  ReadonlyColumnIterator begin_across() const {
    return ReadonlyColumnIterator(this, 0);
  }

  ReadonlyColumnIterator end_across() const {
    return ReadonlyColumnIterator(this, this->size());
  }

 private:
  absl::AnyInvocable<float(size_t) const> fn_;
};

}  // namespace details

}  // namespace uchen::math

#endif  // UCHEN_MATH_MATRIX_FUNCTIONAL_H