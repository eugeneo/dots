#include "src/convolution.h"

#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/container/inlined_vector.h"

#include "hwy/highway.h"
#include "hwy/print-inl.h"

namespace uchen::convolution::implementation {

namespace hn = ::hwy::HWY_NAMESPACE;

struct ConvolutionDimensions {
  int channels;
  int height;
  int width;
};

ConvolutionDimensions OutputDims(const ConvolutionDimensions& input_dims,
                                 const ConvolutionOptions& options) {
  int columns =
      input_dims.width - options.kernel_width + 1 + options.padding_width * 2;
  int rows = input_dims.height - options.kernel_height + 1 +
             options.padding_height * 2;
  return {
      .channels = options.output_channels, .height = rows, .width = columns};
}

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

  void operator()(float* HWY_RESTRICT output, size_t output_rows,
                  size_t output_columns) const;

 private:
  void ProcessCornerPadding(float* HWY_RESTRICT output, size_t output_rows,
                            size_t output_columns) const;
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
      int channels = options_.input_channels;
      DCHECK_EQ(channels % hn::Lanes(d_), 0)
          << "Should be a multiple of SIMD lanes: " << channels;
      for (int i = 0; i < channels; i += hn::Lanes(d_)) {
        accumulator = hn::MulAdd(
            hn::Load(d_, data_ + i + kernel_element * channels),
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
      right = process(right, idx - col, idx, output_columns - 1, row);
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

  for (int row = 0; row < options_.kernel_height; ++row) {
    int y = row * options_.kernel_width;
    for (int x = 0; x < options_.kernel_width; ++x) {
      size_t idx = y + x;
      if (row >= pad) {
        top = process(top, idx, idx - pad * options_.kernel_width, 0, col);
      }
      if (row < options_.kernel_height - pad) {
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

  size_t read_row = is_top ? 0 : row_pad;
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
            (out_row * padded_columns + out_col) * options_.output_channels +
            index_;
        output[out_index] = value;
      }
    }
  }
}

template <typename D, typename Loader>
void Kernel<D, Loader>::operator()(float* HWY_RESTRICT output,
                                   size_t output_rows,
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
      output[(row_start + out_left) * options_.output_channels + index_] = left;

      const size_t out_right =
          options_.padding_width + output_columns + (col - 1);
      output[(row_start + out_right) * options_.output_channels + index_] =
          right;
    }
  }

  // Vertical paddings (top + bottom)
  for (size_t pad = 1; pad <= options_.padding_height; ++pad) {
    for (size_t col = 0; col < output_columns; ++col) {
      auto [top, bottom] = TopBottomPad(pad, col, output_rows);
      const size_t col_index = options_.padding_width + col;

      size_t out_top =
          (options_.padding_height - pad) * padded_columns + col_index;
      output[out_top * options_.output_channels + index_] = top;

      size_t out_bottom =
          (options_.padding_height + output_rows + (pad - 1)) * padded_columns +
          col_index;
      output[out_bottom * options_.output_channels + index_] = bottom;
    }
  }

  // --- Corners ---
  ProcessCornerPadding(output, output_rows, output_columns);

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
      size_t ind = (col + row_start) * options_.output_channels + index_;
      output[ind] = hn::GetLane(hn::SumOfLanes(d_, acc));
    }
  }
}

template <typename D, typename V = hn::VFromD<D>>
V WeightGradientsScanLoop(D d, const float* HWY_RESTRICT input,
                          const ConvolutionDimensions& input_dims,
                          const float* HWY_RESTRICT output_gradients,
                          const ConvolutionDimensions& output_dims,
                          int output_channel, int channel, int x, int y,
                          const ConvolutionOptions& options) {
  V accum = hn::Zero(d);
  int min_row = std::max(0, options.padding_height - y);
  int max_row = std::min(output_dims.height,
                         input_dims.height + options.padding_height - y);
  int min_col = std::max(0, options.padding_width - x);
  int max_col =
      std::min(output_dims.width, input_dims.width + options.padding_width - x);
  const float* HWY_RESTRICT output_gradient_row =
      output_gradients + output_channel +
      min_row * output_dims.width * output_dims.channels;
  size_t input_first_row = y - options.padding_height + min_row;
  size_t input_first_column = x - options.padding_width + min_col;
  const float* HWY_RESTRICT base =
      input + channel +
      (input_first_row * input_dims.width + input_first_column) *
          input_dims.channels;
  for (int row = min_row; row < max_row; ++row) {
    const float* HWY_RESTRICT row_base = base;
    for (int col = min_col; col < max_col; ++col) {
      std::ptrdiff_t oi = col * options.output_channels;
      V inp = hn::Load(d, row_base);
      accum = hn::MulAdd(hn::Set(d, output_gradient_row[oi]), inp, accum);
      row_base += input_dims.channels;
    }
    output_gradient_row += output_dims.width * output_dims.channels;
    base += input_dims.width * input_dims.channels;
  }
  return accum;
}

template <typename D, typename V = hn::VFromD<D>>
V InputGradients(D d, const float* HWY_RESTRICT output_gradients,
                 const float* HWY_RESTRICT parameters,
                 const ConvolutionDimensions& input_dims, int column, int row,
                 int channel, const ConvolutionOptions& options) {
  int output_cols =
      input_dims.width - options.kernel_width + 1 + options.padding_width * 2;
  int output_rows = input_dims.height - options.kernel_height + 1 +
                    options.padding_height * 2;
  const int min_x =
      std::max(0, column + options.padding_width - (output_cols - 1));
  const int max_x =
      std::min(options.kernel_width, column + options.padding_width + 1);

  const int min_y =
      std::max(0, row + options.padding_height - (output_rows - 1));
  const int max_y =
      std::min(options.kernel_height, row + options.padding_height + 1);

  size_t output_base = (((row + options.padding_height - min_y) * output_cols) +
                        column + options.padding_width - min_x) *
                       options.output_channels;
  size_t kernel_first_element_offset =
      (+min_y * options.kernel_width + min_x) * options.input_channels +
      channel;

  V v = hn::Zero(d);
  for (int output_channel = 0; output_channel < options.output_channels;
       ++output_channel) {
    size_t kernel_data_index =
        kernel_first_element_offset + output_channel * options.kernel_height *
                                          options.kernel_width *
                                          options.input_channels;
    size_t output_el = output_base + output_channel;
    for (int y = min_y; y < max_y; ++y) {
      for (int x = min_x; x < max_x; ++x) {
        v = hn::MulAdd(hn::Set(d, output_gradients[output_el]),
                       hn::Load(d, parameters + kernel_data_index), v);
        output_el -= options.output_channels;
        kernel_data_index += options.input_channels;
      }
    }
  }
  return v;
}

}  // namespace

// Compiles per SIMD target.
template <size_t Channels>
HWY_ATTR void Conv2dHighway(std::span<const float> input,
                            std::span<float> output,
                            std::span<const float> weights, size_t columns,
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
      read_offsets.push_back((col + row * columns) * options.input_channels);
    }
  }
  size_t rows = input.size() / columns / options.input_channels;
  DataLoader<D, Channels> loader(d, input, read_offsets, columns,
                                 options.input_channels);
  for (size_t kernel = 0; kernel < options.output_channels; ++kernel) {
    Kernel k(
        d,
        weights.subspan(kernel * read_offsets.size() * options.input_channels,
                        read_offsets.size() * options.input_channels),
        kernel, loader, options);
    k(output.data(), rows - options.kernel_height + 1,
      columns - options.kernel_width + 1);
  }
}

HWY_ATTR void ParameterGradientsHighway(
    const float* HWY_RESTRICT output_gradients, const float* HWY_RESTRICT input,
    float* HWY_RESTRICT out_parameter_gradient,
    const ConvolutionDimensions& input_dims,
    const ConvolutionOptions& options) {
  using D = hn::FixedTag<float, 4>;
  using V = hn::VFromD<D>;
  D d;
  CHECK_EQ(options.input_channels % hn::Lanes(d), 0)
      << options.input_channels << " can't be mapped to " << hn::Lanes(d);
  ConvolutionDimensions output_dims = OutputDims(input_dims, options);
  const size_t kernel_elements =
      options.input_channels * options.kernel_height * options.kernel_width;
  for (int output_channel = 0; output_channel < options.output_channels;
       ++output_channel) {
    float* HWY_RESTRICT kernel_element =
        out_parameter_gradient + output_channel * kernel_elements;
    for (int y = 0; y < options.kernel_height; ++y) {
      for (int x = 0; x < options.kernel_width; ++x) {
        const std::ptrdiff_t kernel_xy_offset =
            (y * options.kernel_width + x) * options.input_channels;
        for (int channel = 0; channel < options.input_channels;
             channel += hn::Lanes(d)) {
          V accum = WeightGradientsScanLoop(
              d, input, input_dims, output_gradients, output_dims,
              output_channel, channel, x, y, options);
          hn::Store(accum, d, kernel_element + channel + kernel_xy_offset);
        }
      }
    }
  }
}

HWY_ATTR void InputGradientsHighway(const float* HWY_RESTRICT output_gradients,
                                    const float* HWY_RESTRICT parameters,
                                    float* HWY_RESTRICT out_input_gradients,
                                    const ConvolutionDimensions& input_dims,
                                    const ConvolutionOptions& options) {
  using D = hn::FixedTag<float, 4>;
  using V = hn::VFromD<D>;
  D d;
  CHECK_EQ(options.input_channels % hn::Lanes(d), 0)
      << "Number of input channels should be a multiple of " << hn::Lanes(d);
  float* HWY_RESTRICT write_ptr = out_input_gradients;
  for (int row = 0; row < input_dims.height; ++row) {
    for (int column = 0; column < input_dims.width; ++column) {
      for (int channel = 0; channel < options.input_channels;
           channel += hn::Lanes(d)) {
        V grad = InputGradients(d, output_gradients, parameters, input_dims,
                                column, row, channel, options);
        hn::Store(grad, d, write_ptr);
        write_ptr += hn::Lanes(d);
      }
    }
  }
}

void ReluHighway(float* HWY_RESTRICT data, size_t len) {
  using D = hn::ScalableTag<float>;
  using V = hn::VFromD<D>;
  D d;
  CHECK(hn::IsAligned(d, data));
  V zero = hn::Zero(d);
  const size_t vec_end = len & ~(hn::Lanes(d) - 1);  // rounds down
  for (size_t index = 0; index < vec_end; index += hn::Lanes(d)) {
    V v = hn::Load(d, data + index);
    v = hn::Max(zero, v);
    hn::Store(v, d, data + index);
  }
  for (size_t index = vec_end; index < len; ++index) {
    data[index] = std::max(data[index], 0.f);
  }
}

}  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

void Conv2d(std::span<const float> input, std::span<float> output,
            std::span<const float> weights, int columns,
            const ConvolutionOptions& options) {
  std::fill(output.begin(), output.end(), 0);

  int rows = input.size() / options.input_channels / columns;
  ConvolutionDimensions out_dims = OutputDims(
      {.channels = options.input_channels, .height = rows, .width = columns},
      options);

  CHECK_GE(output.size(),
           options.output_channels * out_dims.height * out_dims.width);
  CHECK_EQ(options.input_channels % 4,
           0);  // Can't do SIMD otherwise. Just pad the input with zeroes
  // Here we have an opportunity to do some special cases.
  if (options.input_channels == 4) {
    HWY_STATIC_DISPATCH(Conv2dHighway<4>)(input, output, weights, columns,
                                          options);
  } else {
    // Will use dynamic channels count.
    HWY_STATIC_DISPATCH(Conv2dHighway<0>)(input, output, weights, columns,
                                          options);
  }
}

void Conv2dParameterGradients(std::span<const float> output_gradients,
                              std::span<const float> input,
                              std::span<float> out_parameter_gradient,
                              int input_columns,
                              const ConvolutionOptions& options) {
  std::fill(out_parameter_gradient.begin(), out_parameter_gradient.end(), 0);
  const int input_rows = input.size() / options.input_channels / input_columns;
  ConvolutionDimensions input_dims = {.channels = options.input_channels,
                                      .height = input_rows,
                                      .width = input_columns};

  ConvolutionDimensions output_dims = OutputDims(input_dims, options);

  CHECK_EQ(output_gradients.size(),
           output_dims.width * output_dims.height * options.output_channels);
  CHECK_EQ(out_parameter_gradient.size(),
           options.input_channels * options.output_channels *
               options.kernel_height * options.kernel_width);
  HWY_STATIC_DISPATCH(ParameterGradientsHighway)(
      output_gradients.data(), input.data(), out_parameter_gradient.data(),
      input_dims, options);
}

void Conv2dInputGradients(std::span<const float> output_gradients,
                          std::span<const float> parameters,
                          std::span<float> out_input_gradients,
                          int input_columns,
                          const ConvolutionOptions& options) {
  CHECK_EQ(parameters.size(), options.output_channels * options.input_channels *
                                  options.kernel_height * options.kernel_width);
  CHECK_EQ(
      out_input_gradients.size() % (input_columns * options.input_channels), 0);
  std::fill(out_input_gradients.begin(), out_input_gradients.end(), 0);
  int input_rows =
      out_input_gradients.size() / input_columns / options.input_channels;
  CHECK_EQ(output_gradients.size(), options.output_channels *
                                        (input_columns - options.kernel_width +
                                         1 + options.padding_width * 2) *
                                        (input_rows - options.kernel_height +
                                         1 + options.padding_height * 2));
  HWY_STATIC_DISPATCH(InputGradientsHighway)(
      output_gradients.data(), parameters.data(), out_input_gradients.data(),
      {.channels = options.input_channels,
       .height = input_rows,
       .width = input_columns},
      options);
}

void Relu(std::span<float> data) {
  HWY_STATIC_DISPATCH(ReluHighway(data.data(), data.size()));
}

}  // namespace uchen::convolution::implementation