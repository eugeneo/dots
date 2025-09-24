#include "src/convolution.h"

#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/container/inlined_vector.h"

#include "hwy/highway.h"

namespace uchen::convolution::implementation {

namespace hn = ::hwy::HWY_NAMESPACE;

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {
namespace {

// Future directions: focus on reducing RAM traffic. Maximize register usage to
// keep kernels. i.e. 3x4 kernels with 4 channels can stay in registers
// everywhere except old SSE (and possibly WASM).
//
// Another approach would be to keep up to 32 channels in registers while
// scanning through the input. That scan would be cache friendly if the input
// tensor was stored CWH (as Uchen will be doing).
template <typename D, typename Loader>
class Kernel {
 public:
  Kernel(D d, std::span<const float> data, uint32_t index, Loader loader,
         ConvolutionOptions options)
      : d_(d),
        data_(data.data()),
        index_(index),
        loader_(std::move(loader)),
        options_(std::move(options)) {}

  void operator()(float* HWY_RESTRICT output, size_t output_channels,
                  size_t output_rows, size_t output_columns) const;

 private:
  void ProcessCornerPadding(float* HWY_RESTRICT output, size_t output_channels,
                            size_t output_rows, size_t output_columns) const;
  std::pair<float, float> LeftRightPad(int col, size_t row,
                                       size_t output_columns) const;
  std::pair<float, float> TopBottomPad(size_t pad, size_t col,
                                       size_t output_rows) const;
  float ComputeCornerPadding(size_t row_pad, size_t col_pad, bool is_top,
                             bool is_left, size_t output_rows,
                             size_t output_columns) const;

  hn::VFromD<D> process(hn::VFromD<D> accumulator, int kernel_element,
                        int data_element, int data_row, int data_column) const {
    if constexpr (Loader::kChannels != 0) {
      for (size_t i = 0; i < Loader::kChannels; i += hn::Lanes(d_)) {
        accumulator = hn::MulAdd(
            hn::Load(d_, data_ + i + kernel_element * Loader::kChannels),
            loader_.load(data_row, data_column, data_element, i), accumulator);
      }
    } else {
      int channels = loader_.channels();
      DCHECK_EQ(channels % hn::Lanes(d_), 0)
          << "Should be a multiple of SIMD lanes: " << channels;
      for (size_t i = 0; i < channels; i += hn::Lanes(d_)) {
        accumulator = hn::MulAdd(
            hn::LoadU(d_, data_ + i + kernel_element * channels),
            loader_.load(data_row, data_column, data_element, i), accumulator);
      }
    }
    return accumulator;
  }

  D d_;
  const float* HWY_RESTRICT data_;
  uint32_t index_;
  Loader loader_;
  ConvolutionOptions options_;
};

template <typename D, int Channels>
  requires(Channels % 4 == 0)
class DataLoader {
 public:
  static constexpr size_t kChannels = Channels;

  DataLoader(D d, std::span<const float> data,
             std::span<const std::ptrdiff_t> offsets, size_t columns,
             int channels)
      : d_(d),
        data_(data.data()),
        offsets_(offsets),
        columns_(columns),
        channels_(channels) {}

  hn::VFromD<D> load(uint32_t row, uint32_t column, uint32_t element,
                     int channel) const {
    size_t index = column + row * columns_;
    return hn::Load(d_,
                    data_ + index * channels() + offsets_[element] + channel);
  }

  constexpr size_t channels() const {
    if constexpr (Channels == 0) {
      return channels_;
    } else {
      return Channels;
    }
  }

  size_t columns() const { return columns_; }

 private:
  D d_;
  const float* HWY_RESTRICT data_;
  std::span<const std::ptrdiff_t> offsets_;
  size_t columns_;
  int channels_;
};

template <typename D, typename Loader>
std::pair<float, float> Kernel<D, Loader>::LeftRightPad(
    int col, size_t row, size_t output_columns) const {
  using V = hn::VFromD<D>;
  V left = hn::Zero(d_);
  V right = hn::Zero(d_);
  for (int y = 0; y < options_.kernel_height * options_.kernel_width;
       y += options_.kernel_width) {
    // LEFT: taps x in [col, K)
    for (int x = col; x < options_.kernel_width; ++x) {
      const size_t idx = y + x;
      left = process(left, idx, idx - col, 0, row);
    }
    // RIGHT: taps x in [0, K-col), idx = y + x + col
    for (int x = col; x < options_.kernel_width; ++x) {
      const size_t idx = y + x;
      right = process(right, idx, idx, output_columns - 1, row);
    }
  }
  return {hn::GetLane(hn::SumOfLanes(d_, left)),
          hn::GetLane(hn::SumOfLanes(d_, right))};
}

template <typename D, typename Loader>
std::pair<float, float> Kernel<D, Loader>::TopBottomPad(
    size_t pad, size_t col, size_t output_rows) const {
  using V = hn::VFromD<D>;
  V top = hn::Zero(d_);
  V bottom = hn::Zero(d_);

  for (int y = 0; y < options_.kernel_height * options_.kernel_width;
       y += options_.kernel_width) {
    for (int x = 0; x < options_.kernel_width; ++x) {
      size_t idx = y + x;
      if (y / options_.kernel_height >= pad) {
        top = process(top, idx, idx - pad * options_.kernel_width, 0, col);
      }
      if (y / options_.kernel_width < options_.kernel_height - pad) {
        bottom = process(bottom, idx, idx, output_rows - 1, col);
      }
    }
  }

  return {hn::GetLane(hn::SumOfLanes(d_, top)),
          hn::GetLane(hn::SumOfLanes(d_, bottom))};
}

template <typename D, typename Loader>
float Kernel<D, Loader>::ComputeCornerPadding(size_t row_pad, size_t col_pad,
                                              bool is_top, bool is_left,
                                              size_t output_rows,
                                              size_t output_columns) const {
  using V = hn::VFromD<D>;
  V acc = hn::Zero(d_);

  size_t read_row =
      is_top ? 0 : options_.kernel_height + row_pad - options_.kernel_height;
  size_t first_row = (is_top ? row_pad : 0) * options_.kernel_width;
  size_t last_row =
      (options_.kernel_height - (is_top ? 0 : row_pad)) * options_.kernel_width;

  size_t read_col =
      is_left ? 0 : (loader_.columns() + col_pad - options_.kernel_width);
  size_t first_column = is_left ? col_pad : 0;
  size_t last_column = options_.kernel_width - col_pad + first_column;

  for (size_t r = first_row; r < last_row; r += options_.kernel_width) {
    for (size_t c = first_column; c < last_column; ++c) {
      size_t index = c + r;
      size_t read_index = index - (first_column + first_row);
      acc = process(acc, index, read_index, read_row, read_col);
    }
  }
  return hn::GetLane(hn::SumOfLanes(d_, acc));
}

template <typename D, typename Loader>
void Kernel<D, Loader>::ProcessCornerPadding(float* HWY_RESTRICT output,
                                             size_t output_channels,
                                             size_t output_rows,
                                             size_t output_columns) const {
  const size_t padded_columns = output_columns + 2 * options_.padding_width;

  // Kinda tempted to make compile time - but the performance win will be
  // miniscule while binary bloat is real...
  constexpr std::array<std::pair<bool, bool>, 4> corners = {{
      {true, true},   // top-left
      {true, false},  // top-right
      {false, true},  // bottom-left
      {false, false}  // bottom-right
  }};

  // Undecided on order below. Doesn't matter.
  for (const auto& [is_top, is_left] : corners) {
    for (size_t row_pad = 1; row_pad <= options_.padding_height; ++row_pad) {
      for (size_t col_pad = 1; col_pad <= options_.padding_width; ++col_pad) {
        float value = ComputeCornerPadding(row_pad, col_pad, is_top, is_left,
                                           output_rows, output_columns);

        size_t out_row =
            is_top ? (options_.padding_height - row_pad)
                   : (options_.padding_height + output_rows + row_pad - 1);
        size_t out_col =
            is_left ? (options_.padding_width - col_pad)
                    : (options_.padding_width + output_columns + col_pad - 1);

        size_t out_index =
            (out_row * padded_columns + out_col) * output_channels + index_;
        output[out_index] = value;
      }
    }
  }
}

template <typename D, typename Loader>
void Kernel<D, Loader>::operator()(float* HWY_RESTRICT output,
                                   size_t output_channels, size_t output_rows,
                                   size_t output_columns) const {
  using V = hn::VFromD<D>;
  const size_t padded_columns =
      output_columns + options_.padding_width + options_.padding_width;

  // Horizontal paddings (left + right) in one pass
  for (size_t col = 1; col <= options_.padding_width; ++col) {
    for (size_t row = 0; row < output_rows; ++row) {
      auto [left, right] = LeftRightPad(col, row, output_columns);

      const size_t row_start = (row + options_.padding_height) * padded_columns;

      const size_t out_left = options_.padding_width - col;
      output[(row_start + out_left) * output_channels + index_] = left;

      const size_t out_right =
          options_.padding_width + output_columns + (col - 1);
      output[(row_start + out_right) * output_channels + index_] = right;
    }
  }

  // Vertical paddings (top + bottom)
  for (size_t pad = 1; pad <= options_.padding_height; ++pad) {
    for (size_t col = 0; col < output_columns; ++col) {
      auto [top, bottom] = TopBottomPad(pad, col, output_rows);
      const size_t col_index = options_.padding_width + col;

      size_t out_top =
          (options_.padding_height - pad) * padded_columns + col_index;
      output[out_top * output_channels + index_] = top;

      size_t out_bottom =
          (options_.padding_height + output_rows + (pad - 1)) * padded_columns +
          col_index;
      output[out_bottom * output_channels + index_] = bottom;
    }
  }

  // --- Corners ---
  ProcessCornerPadding(output, output_channels, output_rows, output_columns);

  const size_t kernel_elements = options_.kernel_height * options_.kernel_width;

  // Main area
  for (size_t row = 0; row < output_rows; ++row) {
    const size_t row_start = (row + options_.padding_height) * padded_columns +
                             options_.padding_width;
    for (size_t col = 0; col < output_columns; ++col) {
      V acc = hn::Zero(d_);
      for (size_t el = 0; el < kernel_elements; ++el) {
        acc = process(acc, el, el, row, col);
      }
      // This write is not efficient - but speeds up reads from other model
      // layers. Write here is one time but next convolution layer will read
      // this many times - this layout is cache friendly.
      size_t ind = (col + row_start) * output_channels + index_;
      output[ind] = hn::GetLane(hn::SumOfLanes(d_, acc));
    }
  }
}

}  // namespace

// Compiles per SIMD target.
template <size_t Channels>
HWY_ATTR void Conv2dHighway(std::span<const float> input,
                            std::span<float> output,
                            std::span<const float> weights, size_t columns,
                            size_t input_channels, size_t output_channels,
                            const ConvolutionOptions& options) {
  // 4 channels - fixed tag. Will see if can use scalable for more channels.
  using D = hn::FixedTag<float, 4>;
  D d;
  // No restrictions on the output - it's scalar writes
  CHECK(hn::IsAligned(d, input.data()));
  CHECK(hn::IsAligned(d, weights.data()));
  absl::InlinedVector<std::ptrdiff_t, 64> read_offsets;
  for (size_t row = 0; row < options.kernel_height; ++row) {
    for (size_t col = 0; col < options.kernel_width; ++col) {
      read_offsets.push_back((col + row * columns) * input_channels);
    }
  }
  size_t rows = input.size() / columns / input_channels;
  DataLoader<D, Channels> loader(d, input, read_offsets, columns,
                                 input_channels);
  for (size_t kernel = 0; kernel < output_channels; ++kernel) {
    Kernel k(d,
             weights.subspan(kernel * read_offsets.size() * input_channels,
                             read_offsets.size() * input_channels),
             kernel, loader, options);
    k(output.data(), output_channels, rows - options.kernel_height + 1,
      columns - options.kernel_width + 1);
  }
}

}  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

void Conv2d(std::span<const float> input, std::span<float> output,
            std::span<const float> weights, size_t columns,
            size_t input_channels, size_t output_channels,
            const ConvolutionOptions& options) {
  size_t rows = input.size() / input_channels / columns;
  size_t output_columns =
      columns + 1 + options.padding_width * 2 - options.kernel_width;
  size_t output_rows =
      rows + 1 + options.padding_height * 2 - options.kernel_height;

  CHECK_GE(output.size(), output_channels * output_columns * output_rows);
  CHECK_EQ(input_channels % 4,
           0);  // Can't do SIMD otherwise. Just pad the input with zeroes
  // Here we have an opportunity to do some special cases.
  if (input_channels == 4) {
    HWY_STATIC_DISPATCH(Conv2dHighway<4>)(input, output, weights, columns,
                                          input_channels, output_channels,
                                          options);
  } else {
    // Will use dynamic channels count.
    HWY_STATIC_DISPATCH(Conv2dHighway<0>)(input, output, weights, columns,
                                          input_channels, output_channels,
                                          options);
  }
}

}  // namespace uchen::convolution::implementation