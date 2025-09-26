#include "uchen/tensor/tensor.h"

#include <cstddef>

#include "absl/log/check.h"

namespace uchen::tensor {

namespace {

Layout::MemoryLayoutInfo GetMemoryLayoutInfo(const ColumnMajor& cm) {
  const auto [rows, columns] = cm.dims();
  return {.fast_dim_size = rows, .slow_dim_size = columns};
}

Layout::MemoryLayoutInfo GetMemoryLayoutInfo(const RowMajor& rm) {
  const auto [rows, columns] = rm.dims();
  return {.fast_dim_size = columns, .slow_dim_size = rows};
}

}  // namespace

size_t Layout::ArrayIndex(absl::Span<const size_t> dimensions,
                          absl::Span<const size_t> index) const {
  CHECK_EQ(dimensions.size(), index.size());
  size_t matrix_id = 0;
  for (size_t i = 0; i < dimensions.size() - 2; ++i) {
    matrix_id *= dimensions[i];
    matrix_id += index[i];
  }
  matrix_id *= dimensions[dimensions.size() - 2] * dimensions.back();
  return matrix_id + visit([&](const auto& layout) {
           return layout.ToArrayIndex(index[index.size() - 2], index.back());
         });
}

Layout::MemoryLayoutInfo Layout::memory_layout_info() const {
  return std::visit(
      [](const auto& layout) { return GetMemoryLayoutInfo(layout); },
      matrix_layout_);
}

}  // namespace uchen::tensor