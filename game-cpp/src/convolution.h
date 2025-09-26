#ifndef UCHEN_CONVOLUTION_H_
#define UCHEN_CONVOLUTION_H_

#include <array>
#include <cstddef>
#include <span>

#include "uchen/model.h"

namespace uchen::convolution {

namespace implementation {

struct ConvolutionOptions {
  int input_channels;
  int output_channels;
  int padding_height = 0;
  int padding_width = 0;
  int kernel_height = 3;
  int kernel_width = 3;
};

void Conv2d(std::span<const float> input, std::span<float> output,
            std::span<const float> weights, int columns,
            const ConvolutionOptions& options);

void Conv2dParameterGradients(std::span<const float> output_gradients,
                              std::span<const float> input,
                              std::span<float> out_parameter_gradient,
                              int input_columns,
                              const ConvolutionOptions& options);
void Conv2dInputGradients(std::span<const float> output_gradients,
                          std::span<const float> parameters,
                          std::span<float> out_input_gradients,
                          int input_columns, const ConvolutionOptions& options);

}  // namespace implementation

/*
 * This is a streamlined backport of Tensor from future uchen-core. It only
 * supports what is needed for this example.
 *
 * Channels must be multiple of 4 for SIMD optimizations. Just ignore the ones
 * you don't use (set to zero).
 *
 * Storage is CHW (Channels, Height, Width) for better memory access patterns.
 */
template <size_t C, size_t H, size_t W>
  requires(C > 0 && H > 0 && W > 0 && (C % 4 == 0))
class ConvolutionInput {
 public:
  static constexpr size_t channels = C;
  static constexpr size_t height = H;
  static constexpr size_t width = W;

  std::span<const float> data() const { return data_; }

  std::span<float> data() { return data_; }

  friend ConvolutionInput Emancipate(const ConvolutionInput& input) {
    return input;
  }

 private:
  alignas(16) std::array<float, C * H * W> data_;
};

template <typename Input, size_t OutputChannels, size_t KernelHeight,
          size_t KernelWidth>
  requires(OutputChannels % 4 == 0 && KernelHeight > 0 && KernelWidth > 0)
class Conv2dLayer {
 public:
  using input_t = Input;

  template <typename... Args>
  ConvolutionInput<OutputChannels, Input::height, Input::width> operator()(
      const Input& input, const auto& parameters, auto* ctx) const {
    // Simplest case
    if constexpr (Input::channels == 4 && KernelHeight * KernelWidth >= 8) {
      implementation::Conv2d(input.data(), ctx->GetScratchArea()->data(),
                             parameters, Input::width, kOptions);
    } else {
      LOG(FATAL) << "Only 4-channel input is supported";
    }
    return *ctx->GetScratchArea();
  }

 private:
  static constexpr implementation::ConvolutionOptions kOptions = {
      .input_channels = Input::channels,
      .output_channels = OutputChannels,
      .padding_height = 1,
      .padding_width = 1,
      .kernel_height = KernelHeight,
      .kernel_width = KernelWidth,
  };
};

template <size_t OutputChannels, size_t KernelHeight, size_t KernelWidth>
class Conv2dLayerDesc {
 public:
  template <typename Layer>
  constexpr auto stack(const Layer& /* layer */) const {
    return Conv2dLayer<typename Layer::output_t, OutputChannels, KernelHeight,
                       KernelWidth>();
  }
};

template <size_t OutputChannels, size_t KernelHeight = 3,
          size_t KernelWidth = KernelHeight>
static constexpr Layer Conv2d =
    Layer<Conv2dLayerDesc<OutputChannels, KernelHeight, KernelWidth>>();

template <typename I, size_t OC, size_t KernelHeight, size_t KernelWidth>
auto ParameterProvider(
    const Conv2dLayer<I, OC, KernelHeight, KernelWidth>& layer,
    std::span<const float> data, std::shared_ptr<memory::Deletable> ref) {
  return Parameters<OC * KernelHeight * KernelWidth * I::channels>(
      data, std::move(ref));
}

}  // namespace uchen::convolution

template <size_t channels, size_t height, size_t width, size_t OutputChannels,
          size_t KernelHeight, size_t KernelWidth>
struct uchen::LayerTraits<
    uchen::convolution::Conv2dLayer<
        uchen::convolution::ConvolutionInput<channels, height, width>,
        OutputChannels, KernelHeight, KernelWidth>,
    uchen::convolution::ConvolutionInput<channels, height, width>>
    : public LayerTraitFields<
          uchen::convolution::ConvolutionInput<OutputChannels, height, width>,
          KernelHeight * KernelWidth * channels * OutputChannels,
          uchen::convolution::ConvolutionInput<OutputChannels, height, width>> {
};

#endif  // UCHEN_CONVOLUTION_H_