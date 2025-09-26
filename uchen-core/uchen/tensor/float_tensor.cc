#include "uchen/tensor/float_tensor.h"

#include <cstddef>
#include <cstdlib>

#include "absl/log/check.h"

namespace uchen::core {

namespace details {

size_t DimSliceTranslator::dim(size_t dim, const BasicTensor& input) const {
  DCHECK(dim < input.rank());
  return dim == dim_ ? size_ : input.dim(dim);
}

size_t DimSliceTranslator::Translate(const size_t index,
                                     const BasicTensor& input) const {
  size_t current_rank = input.rank() - 1;
  size_t retained = 1;
  for (; current_rank > dim_; --current_rank) {
    retained *= input.dim(current_rank);
  }
  size_t high = index / retained;
  return index % retained +
         (high % size_ + start_ + high / size_ * input.dim(dim_)) * retained;
}

size_t TransposeTranslator::dim(size_t dim, const BasicTensor& input) const {
  DCHECK(dim < input.rank());
  const size_t rank = input.rank();
  if (dim < rank - 2) {
    return input.dim(dim);
  } else if (dim == rank - 2) {
    return input.dim(rank - 1);
  } else {
    return input.dim(rank - 2);
  }
}

size_t TransposeTranslator::Translate(size_t index,
                                      const BasicTensor& input) const {
  const size_t rank = input.rank();
  DCHECK(rank >= 2);
  size_t cols = dim(rank - 1, input);
  size_t rows = dim(rank - 2, input);
  size_t matrix = cols * rows;
  size_t high = index / matrix;
  size_t low = index % matrix;
  size_t row = low / cols;
  size_t col = low % cols;
  return high * matrix + col * rows + row;
}

}  // namespace details
}  // namespace uchen::core