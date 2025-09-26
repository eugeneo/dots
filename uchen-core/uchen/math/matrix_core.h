#ifndef UCHEN_MATH_MATRIX_CORE_H
#define UCHEN_MATH_MATRIX_CORE_H

#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace uchen::math {

class MatrixBase {
 public:
  virtual ~MatrixBase() = default;
  virtual uint32_t rows() const = 0;
  virtual uint32_t columns() const = 0;
  virtual float GetRowMajor(size_t index) const = 0;
  virtual float GetColumnMajor(size_t index) const = 0;
};

namespace details {

constexpr size_t ConvertIndexDirection(size_t index, uint32_t Source,
                                       uint32_t Dest) {
  return index / Source + index % Source * Dest;
}

template <typename S>
struct ElementType {
  using type = typename S::element_type;
};

template <typename S, size_t N>
struct ElementType<std::array<S, N>> {
  // Can use S but this seems more explicit.
  using type = std::array<S, N>::value_type;
};

template <typename Self>
class IteratorBase {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = float;

  IteratorBase() = default;
  explicit IteratorBase(size_t index) : index_(index) {}

  Self& operator++() {
    ++index_;
    return *static_cast<Self*>(this);
  }

  Self operator++(int) {
    Self copy = *static_cast<Self*>(this);
    ++index_;
    return copy;
  }

  Self& operator--() {
    --index_;
    return *static_cast<Self*>(this);
  }

  Self operator--(int) {
    Self copy = *static_cast<Self*>(this);
    --index_;
    return copy;
  }

  std::weak_ordering operator<=>(const Self& other) const {
    return index_ <=> other.index_;
  }

  bool operator==(const IteratorBase& other) const = default;
  bool operator!=(const IteratorBase& other) const = default;

  Self operator+(difference_type n) const {
    Self copy = *static_cast<const Self*>(this);
    copy.index_ += n;
    return copy;
  }

 protected:
  size_t index_ = 0;
};

class ReadonlyRowIterator : public IteratorBase<ReadonlyRowIterator> {
 public:
  ReadonlyRowIterator() = default;
  ReadonlyRowIterator(const MatrixBase* matrix, size_t index)
      : IteratorBase(index), matrix_(matrix) {}

  float operator*() const { return matrix_->GetRowMajor(index_); }

 private:
  const MatrixBase* matrix_;
};

class ReadonlyColumnIterator : public IteratorBase<ReadonlyColumnIterator> {
 public:
  ReadonlyColumnIterator() = default;
  ReadonlyColumnIterator(const MatrixBase* matrix, size_t index)
      : IteratorBase(index), matrix_(matrix) {}

  float operator*() const { return matrix_->GetColumnMajor(index_); }

 private:
  const MatrixBase* matrix_;
};

class WritableCrossIterator : public IteratorBase<WritableCrossIterator> {
 public:
  WritableCrossIterator() = default;
  WritableCrossIterator(uint32_t source, uint32_t destination,
                        std::span<float> data, size_t index)
      : IteratorBase(index),
        source_(source),
        destination_(destination),
        data_(data) {}

  float& operator*() {
    return data_[ConvertIndexDirection(index_, source_, destination_)];
  }

 private:
  size_t source_ = 0;
  size_t destination_ = 0;
  std::span<float> data_;
};

}  // namespace details

template <uint32_t Rows, uint32_t Cols>
  requires(Rows > 0 && Cols > 0)
class MatrixWithSize : public MatrixBase {
 public:
  static constexpr uint32_t R = Rows;
  static constexpr uint32_t C = Cols;

  constexpr uint32_t rows() const final { return Rows; }
  constexpr uint32_t columns() const final { return Cols; }
  static constexpr size_t size() { return static_cast<size_t>(Rows) * Cols; }
};

template <typename S>
struct Transposed {
  using type = S::Transposed;
};

template <uint32_t Rows, uint32_t Cols, typename Store>
  requires(Rows > 0 && Cols > 0)
class RowMajorLayout;

template <uint32_t Rows, uint32_t Cols, typename Store>
  requires(Rows > 0 && Cols > 0)
class ColumnMajorLayout : public MatrixWithSize<Rows, Cols> {
 public:
  static constexpr uint32_t R = Rows;
  static constexpr uint32_t C = Cols;
  using element_type = typename details::ElementType<Store>::type;

  static constexpr size_t size() { return static_cast<size_t>(Rows) * Cols; }

  using Transposed =
      RowMajorLayout<Cols, Rows, std::span<element_type, size()>>;

  template <typename... Args>
  explicit ColumnMajorLayout(Args&&... args)
      : store_(std::forward<Args>(args)...) {}

  constexpr float GetRowMajor(size_t index) const noexcept final {
    return store_[RowMajorIndexToNative(index)];
  }

  float GetColumnMajor(size_t index) const { return store_[index]; }

  element_type& GetRowMajorRef(size_t index) {
    return store_[RowMajorIndexToNative(index)];
  }

  element_type& GetColumnMajorRef(size_t index) { return store_[index]; }

  std::span<const element_type, size()> data_view() const { return store_; }
  std::span<element_type, size()> data_view() { return store_; }

  auto begin() const { return details::ReadonlyRowIterator(this, 0); }
  auto begin() { return Iterator(this, store_, 0); }
  auto end() const { return details::ReadonlyRowIterator(this, size()); }
  auto end() { return Iterator(this, store_, size()); }

  auto begin_across() const { return store_.begin(); }
  auto begin_across() { return store_.begin(); }
  auto end_across() const { return store_.end(); }
  auto end_across() { return store_.end(); }

 protected:
  auto begin_natural() { return store_.begin(); }
  auto end_natural() { return store_.end(); }
  constexpr auto OtherBeginCompatible(
      const std::derived_from<MatrixWithSize<R, C>> auto& other) {
    return other.begin_across();
  }
  constexpr auto OtherEndCompatible(
      const std::derived_from<MatrixWithSize<R, C>> auto& other) {
    return other.end_across();
  }

 private:
  static constexpr details::WritableCrossIterator Iterator(
      MatrixBase* matrix, std::span<float, size()> data, size_t index) {
    return {C, R, data, index};
  }

  static constexpr details::WritableCrossIterator Iterator(
      MatrixBase* matrix, std::array<float, size()>& data, size_t index) {
    return {C, R, data, index};
  }

  static constexpr details::ReadonlyRowIterator Iterator(
      MatrixBase* matrix, std::span<const float> data, size_t index) {
    return {matrix, index};
  }

  constexpr size_t RowMajorIndexToNative(size_t index) const {
    return details::ConvertIndexDirection(index, Cols, Rows);
  }

  constexpr size_t ColumnMajorIndexToNative(size_t index) const {
    return index;
  }

  Store store_;
};

template <uint32_t Rows, uint32_t Cols, typename Store>
  requires(Rows > 0 && Cols > 0)
class RowMajorLayout : public MatrixWithSize<Rows, Cols> {
 public:
  using element_type = typename details::ElementType<Store>::type;
  static constexpr uint32_t R = Rows;
  static constexpr uint32_t C = Cols;

  static constexpr size_t size() { return static_cast<size_t>(Rows) * Cols; }

  using Transposed =
      ColumnMajorLayout<Cols, Rows, std::span<element_type, size()>>;

  template <typename... Args>
  explicit RowMajorLayout(Args&&... args)
      : store_(std::forward<Args>(args)...) {}

  float GetRowMajor(size_t index) const noexcept final { return store_[index]; }

  float GetColumnMajor(size_t index) const {
    return store_[ColumnMajorIndexToNative(index)];
  }

  float& GetRowMajorRef(size_t index) { return store_[index]; }

  float& GetColumnMajorRef(size_t index) {
    return store_[ColumnMajorIndexToNative(index)];
  }

  std::span<const element_type, size()> data_view() const { return store_; }
  std::span<element_type, size()> data_view() { return store_; }

  auto begin() const { return store_.begin(); }
  auto begin() { return store_.begin(); }
  auto end() const { return store_.end(); }
  auto end() { return store_.end(); }

  auto begin_across() const { return details::ReadonlyColumnIterator(this, 0); }
  auto begin_across() { return Iterator(this, store_, 0); }
  auto end_across() const {
    return details::ReadonlyColumnIterator(this, size());
  }
  auto end_across() { return Iterator(this, store_, size()); }

 protected:
  auto begin_natural() { return store_.begin(); }
  auto end_natural() { return store_.end(); }
  constexpr auto OtherBeginCompatible(
      const std::derived_from<MatrixWithSize<R, C>> auto& other) {
    return other.begin();
  }
  constexpr auto OtherEndCompatible(
      const std::derived_from<MatrixWithSize<R, C>> auto& other) {
    return other.end();
  }

 private:
  static constexpr details::WritableCrossIterator Iterator(
      MatrixBase* matrix, std::span<float> data, size_t index) {
    return {R, C, data, index};
  }

  static constexpr details::WritableCrossIterator Iterator(
      MatrixBase* matrix, std::array<float, size()>& data, size_t index) {
    return {R, C, data, index};
  }

  static constexpr details::WritableCrossIterator Iterator(
      MatrixBase* matrix, std::vector<float>& data, size_t index) {
    return {R, C, data, index};
  }

  template <typename C>
    requires(std::convertible_to<C, std::span<const float>> &&
             !std::convertible_to<C, std::span<float>>)
  static constexpr details::ReadonlyRowIterator Iterator(MatrixBase* matrix,
                                                         C& data,
                                                         size_t index) {
    return {matrix, index};
  }

  static constexpr size_t ColumnMajorIndexToNative(size_t index) {
    return details::ConvertIndexDirection(index, Rows, Cols);
  }

  Store store_;
};

}  // namespace uchen::math

#endif  // UCHEN_MATH_MATRIX_CORE_H