#include "src/convolution.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/strings/substitute.h"

#include "hwy/highway.h"

namespace uchen::convolution::implementation {

namespace hn = ::hwy::HWY_NAMESPACE;
HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {
namespace {

// 3x3 kernels can be kept in memory (9 registers). Can't use 16 registers
// on x86 until AVX-512 is a baseline
template <typename D, typename Loader>
class Kernel {
 public:
  static constexpr uint32_t kChannels = 4;

  Kernel(D d, std::span<const float> data, uint32_t index, uint32_t width,
         Loader loader)
      : d_(d),
        data_(data.data()),
        index_(index),
        width_(width),
        height_(data.size() / width_ / kChannels),
        size_(data.size() / kChannels),
        loader_(std::move(loader)) {}

  uint32_t index() const { return index_; }
  uint32_t size() const { return size_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

  int input_rows() const { return loader_.rows(); }
  int input_columns() const { return loader_.columns(); }
  int input_elements() const { return loader_.elements(); }

  hn::VFromD<D> process(hn::VFromD<D> accumulator, int kernel_element,
                        int data_element, int data_row, int data_column) const {
    return hn::MulAdd(hn::Load(d_, data_ + kernel_element * Loader::kChannels),
                      loader_.load(data_row, data_column, data_element),
                      accumulator);
  }

 private:
  D d_;
  const float* HWY_RESTRICT data_;
  uint32_t index_;
  uint32_t width_;
  uint32_t height_;
  uint32_t size_;
  Loader loader_;
};

template <typename D>
class FourChannelDataLoader {
 public:
  static constexpr size_t kChannels = 4;

  FourChannelDataLoader(D d, std::span<const float> data,
                        std::span<const std::ptrdiff_t> offsets, size_t columns)
      : d_(d), data_(data.data()), offsets_(offsets), columns_(columns) {}

  hn::VFromD<D> load(uint32_t row, uint32_t column, uint32_t element) const {
    size_t index = column + row * columns_;
    return hn::Load(d_, data_ + index * 4 + offsets_[element]);
  }

  uint32_t elements() const { return offsets_.size(); }
  size_t columns() const { return columns_; }
  size_t rows() const { return elements() / columns(); }

 private:
  D d_;
  const float* HWY_RESTRICT data_;
  std::span<const std::ptrdiff_t> offsets_;
  size_t columns_;
};

struct AnyChannelsDataLoader {};

template <size_t Channels, typename D>
using Loader = std::conditional_t<Channels == 4, FourChannelDataLoader<D>,
                                  AnyChannelsDataLoader>;

std::pair<float, float> LeftRightPad(const auto& kernel, int col, size_t row,
                                     size_t output_columns) {
  using D = hn::FixedTag<float, 4>;
  using V = hn::VFromD<D>;
  D d;
  V left = hn::Zero(d);
  V right = hn::Zero(d);
  for (int y = 0; y < kernel.size(); y += kernel.width()) {
    // LEFT: taps x in [col, K)
    for (int x = col; x < kernel.width(); ++x) {
      const size_t idx = y + x;
      left = kernel.process(left, idx, idx - col, 0, row);
    }
    // RIGHT: taps x in [0, K-col), idx = y + x + col
    for (int x = col; x < kernel.width(); ++x) {
      const size_t idx = y + x;
      right = kernel.process(right, idx, idx, output_columns - 1, row);
    }
  }
  return {hn::GetLane(hn::SumOfLanes(d, left)),
          hn::GetLane(hn::SumOfLanes(d, right))};
}

std::pair<float, float> TopBottomPad(const auto& kernel, size_t pad, size_t col,
                                     size_t output_rows) {
  using D = hn::FixedTag<float, 4>;
  using V = hn::VFromD<D>;
  D d;
  V top = hn::Zero(d);
  V bottom = hn::Zero(d);

  for (int y = 0; y < kernel.size(); y += kernel.width()) {
    for (int x = 0; x < kernel.width(); ++x) {
      size_t idx = y + x;
      if (y / kernel.height() >= pad) {
        top = kernel.process(top, idx, idx - pad * kernel.width(), 0, col);
      }
      if (y / kernel.width() < kernel.width() - pad) {
        bottom = kernel.process(bottom, idx, idx, output_rows - 1, col);
      }
    }
  }

  return {hn::GetLane(hn::SumOfLanes(d, top)),
          hn::GetLane(hn::SumOfLanes(d, bottom))};
}

float ComputeCornerPadding(const auto& kernel, size_t row_pad, size_t col_pad,
                           bool is_top, bool is_left, size_t output_rows,
                           size_t output_columns,
                           const ConvolutionOptions& options) {
  using D = hn::FixedTag<float, 4>;
  using V = hn::VFromD<D>;
  D d;
  V acc = hn::Zero(d);

  size_t read_row =
      is_top ? 0 : kernel.input_rows() + row_pad - kernel.height();
  size_t first_row = (is_top ? row_pad : 0) * kernel.width();
  size_t last_row = (kernel.height() - (is_top ? 0 : row_pad)) * kernel.width();

  size_t read_col =
      is_left ? 0 : (kernel.input_columns() + col_pad - kernel.width());
  size_t first_column = is_left ? col_pad : 0;
  size_t last_column = kernel.width() - col_pad + first_column;

  for (size_t r = first_row; r < last_row; r += kernel.width()) {
    for (size_t c = first_column; c < last_column; ++c) {
      size_t index = c + r;
      size_t read_index = index - (first_column + first_row);
      acc = kernel.process(acc, index, read_index, read_row, read_col);
    }
  }
  return hn::GetLane(hn::SumOfLanes(d, acc));
}

void ProcessCornerPadding(size_t output_channels, size_t output_rows,
                          size_t output_columns, float* HWY_RESTRICT output,
                          const auto& kernel,
                          const ConvolutionOptions& options) {
  const size_t padded_columns = output_columns + 2 * options.padding_width;

  constexpr std::array<std::pair<bool, bool>, 4> corners = {{
      {true, true},   // top-left
      {true, false},  // top-right
      {false, true},  // bottom-left
      {false, false}  // bottom-right
  }};

  for (size_t row_pad = 1; row_pad <= options.padding_height; ++row_pad) {
    for (size_t col_pad = 1; col_pad <= options.padding_width; ++col_pad) {
      for (const auto& [is_top, is_left] : corners) {
        float value =
            ComputeCornerPadding(kernel, row_pad, col_pad, is_top, is_left,
                                 output_rows, output_columns, options);

        size_t out_row =
            is_top ? (options.padding_height - row_pad)
                   : (options.padding_height + output_rows + row_pad - 1);
        size_t out_col =
            is_left ? (options.padding_width - col_pad)
                    : (options.padding_width + output_columns + col_pad - 1);

        size_t out_index =
            (out_row * padded_columns + out_col) * output_channels +
            kernel.index();
        output[out_index] = value;
      }
    }
  }
}

template <typename D>
void Process4ChannelKernel(D d, size_t output_channels, size_t output_rows,
                           size_t output_columns, float* HWY_RESTRICT output,
                           const auto& kernel,
                           const ConvolutionOptions& options) {
  using V = hn::VFromD<D>;
  const size_t padded_columns =
      output_columns + options.padding_width + options.padding_width;

  // Horizontal paddings (left + right) in one pass
  for (size_t col = 1; col <= options.padding_width; ++col) {
    for (size_t row = 0; row < output_rows; ++row) {
      auto [left, right] = LeftRightPad(kernel, col, row, output_columns);

      const size_t row_start = (row + options.padding_height) * padded_columns;

      const size_t out_left = options.padding_width - col;
      output[(row_start + out_left) * output_channels + kernel.index()] = left;

      const size_t out_right =
          options.padding_width + output_columns + (col - 1);
      output[(row_start + out_right) * output_channels + kernel.index()] =
          right;
    }
  }

  // Vertical paddings (top + bottom)
  for (size_t pad = 1; pad <= options.padding_height; ++pad) {
    for (size_t col = 0; col < output_columns; ++col) {
      auto [top, bottom] = TopBottomPad(kernel, pad, col, output_rows);

      const size_t col_index = options.padding_width + col;

      // top row
      size_t out_top =
          (options.padding_height - pad) * padded_columns + col_index;
      output[out_top * output_channels + kernel.index()] = top;

      // bottom row
      size_t out_bottom =
          (options.padding_height + output_rows + (pad - 1)) * padded_columns +
          col_index;
      output[out_bottom * output_channels + kernel.index()] = bottom;
    }
  }

  // --- Corners ---
  ProcessCornerPadding(output_channels, output_rows, output_columns, output,
                       kernel, options);

  // Main area
  for (size_t row = 0; row < output_rows; ++row) {
    const size_t row_start =
        (row + options.padding_height) * padded_columns + options.padding_width;
    for (size_t col = 0; col < output_columns; ++col) {
      V acc = hn::Zero(d);
      for (size_t el = 0; el < kernel.input_elements(); ++el) {
        acc = kernel.process(acc, el, el, row, col);
      }
      // This write is not efficient - but speeds up future reads. We will write
      // once but next convolution will read this many times - this layout helps
      // with caching
      size_t index = (col + row_start) * output_channels + kernel.index();
      output[index] = hn::GetLane(hn::SumOfLanes(d, acc));
    }
  }
}

}  // namespace

template <size_t Channels>
HWY_ATTR void Conv2dHighway(std::span<const float> input,
                            std::span<float> output,
                            std::span<const float> weights, size_t columns,
                            size_t input_channels, size_t output_channels,
                            const ConvolutionOptions& options) {
  // 4 channels - fixed tag. Will see if can use scalable for more channels.
  using D = hn::FixedTag<float, 4>;
  D d;
  absl::InlinedVector<std::ptrdiff_t, 64> read_offsets;
  for (size_t row = 0; row < options.kernel_height; ++row) {
    for (size_t col = 0; col < options.kernel_width; ++col) {
      read_offsets.push_back((col + row * columns) * input_channels);
    }
  }
  size_t rows = input.size() / columns / input_channels;
  Loader<Channels, D> loader(d, input, read_offsets, columns);
  for (size_t kernel = 0; kernel < output_channels; ++kernel) {
    Kernel k(d,
             weights.subspan(kernel * read_offsets.size() * input_channels,
                             read_offsets.size() * input_channels),
             kernel, options.kernel_width, loader);
    Process4ChannelKernel(d, output_channels, rows - options.kernel_height + 1,
                          columns - options.kernel_width + 1, output.data(), k,
                          options);
  }
}

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

void Conv2d(std::span<const float> input, std::span<float> output,
            std::span<const float> weights, size_t columns,
            size_t input_channels, size_t output_channels,
            const ConvolutionOptions& options) {
  constexpr int channels = 4;
  size_t rows = input.size() / channels / columns;
  DLOG(INFO) << absl::Substitute("Input CxRxC: $0x$1x$2, $3 elements", channels,
                                 rows, columns, input.size());
  size_t output_columns =
      columns - options.kernel_width + 1 + options.padding_width * 2;
  size_t output_rows =
      rows - options.kernel_height + 1 + options.padding_height * 2;
  DLOG(INFO) << absl::Substitute(
      "Input CxRxC: $0x$1x$2, $3 elements", output_channels, output_rows,
      output_columns, output_channels * output_columns * output_rows);

  CHECK_GE(output.size(), output_channels * output_columns * output_rows);
  CHECK_EQ(input_channels % 4, 0);  // Can't do SIMD otherwise
  // Special optimized case
  if (input_channels == 4) {
    HWY_STATIC_DISPATCH(Conv2dHighway<4>)(input, output, weights, columns,
                                          input_channels, output_channels,
                                          options);
  } else {
    LOG(FATAL) << "Not implemented!";
  }
}

}  // namespace uchen::convolution::implementation