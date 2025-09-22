#include "src/convolution.h"

#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/log/log.h"
#include "absl/strings/str_join.h"

#include "hwy/highway.h"

namespace uchen::convolution::implementation {

namespace hn = ::hwy::HWY_NAMESPACE;
HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {
namespace {

// TODO Keep these kernels in SIMD registers - at least if they are under 4x4
template <typename D>
class FourChannelKernel {
 public:
  FourChannelKernel(D d, std::span<const float> data)
      : d_(d), data_(data.data()) {}

  hn::VFromD<D> load(uint32_t index) const {
    return hn::Load(d_, data_ + index * channels());
  }

  constexpr uint32_t channels() const { return 4; }

 private:
  D d_;
  const float* HWY_RESTRICT data_;
};

template <typename D>
class DataLoader {
 public:
  DataLoader(D d, std::span<const float> data,
             std::span<const std::ptrdiff_t> offsets, size_t columns)
      : d_(d), data_(data.data()), offsets_(offsets), columns_(columns) {}

  hn::VFromD<D> load(uint32_t row, uint32_t column, uint32_t element) const {
    return hn::Load(
        d_, data_ + (column + row * columns_) * channels() + offsets_[element]);
  }

  constexpr uint32_t channels() const { return 4; }
  uint32_t elements() const { return offsets_.size(); }

 private:
  D d_;
  const float* HWY_RESTRICT data_;
  std::span<const std::ptrdiff_t> offsets_;
  size_t columns_;
};

template <typename D>
void Process4ChannelKernel(D d, size_t output_channels, size_t output_rows,
                           size_t output_columns, float* HWY_RESTRICT output,
                           const auto& kernel, const auto& input) {
  using V = hn::VFromD<D>;
  for (size_t row = 0; row < output_rows; ++row) {
    for (size_t col = 0; col < output_columns; ++col) {
      V acc = hn::Zero(d);
      for (size_t el = 0; el < input.elements(); ++el) {
        acc = hn::MulAdd(input.load(row, col, el), kernel.load(el), acc);
      }
      // This write is not efficient - but speeds up future reads. We will write
      // once but next convolution will read this many times - this layout helps
      // with caching
      size_t index = col + row * output_columns * output_channels;
      output[index] = hn::GetLane(hn::SumOfLanes(d, acc));
    }
  }
}

}  // namespace

HWY_ATTR void Conv2dStride4(std::span<const float> input,
                            std::span<float> output,
                            std::span<const float> weights, size_t columns,
                            size_t output_channels,
                            const ConvolutionOptions& options) {
  // 4 channels - fixed tag. Will see if can use scalable for more channels.
  hn::FixedTag<float, 4> d;
  constexpr size_t input_channels = 4;
  absl::InlinedVector<std::ptrdiff_t, 64> read_offsets;
  for (size_t row = 0; row < options.kernel_height; ++row) {
    for (size_t col = 0; col < options.kernel_width; ++col) {
      read_offsets.push_back((col + row * columns) * input_channels);
    }
  }
  size_t rows = input.size() / columns / input_channels;
  LOG(INFO) << "Conv2dStride4: read_offsets "
            << absl::StrJoin(read_offsets, ", ") << " Columns " << columns
            << " rows " << rows;
  DataLoader loader(d, input, read_offsets, columns);
  for (size_t kernel = 0; kernel < output_channels; ++kernel) {
    FourChannelKernel k(
        d, weights.subspan(kernel * read_offsets.size() * input_channels,
                           read_offsets.size() * input_channels));
    Process4ChannelKernel(d, output_channels, rows - options.kernel_height + 1,
                          columns - options.kernel_width + 1,
                          output.data() + kernel, k, loader);
  }
}

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

void Conv2dStride4(std::span<const float> input, std::span<float> output,
                   std::span<const float> weights, size_t columns,
                   size_t output_channels, const ConvolutionOptions& options) {
  HWY_STATIC_DISPATCH(Conv2dStride4)(input, output, weights, columns,
                                     output_channels, options);
}

}  // namespace uchen::convolution::implementation