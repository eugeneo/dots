#ifndef UCHEN_CONVOLUTION_H_
#define UCHEN_CONVOLUTION_H_

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

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
void Relu(std::span<float> data);

}  // namespace implementation

template <size_t S>
struct AlignedArray {
  std::array<float, S> data alignas(16);
};

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
  static constexpr size_t elements = C * H * W;

  using store_type_t = AlignedArray<elements>;

  ConvolutionInput() : data_(static_cast<float*>(nullptr), elements) {
    auto store = uchen::memory::ArrayStore<float, elements>::NewInstance();
    data_ = store->data();
    store_ = std::move(store);
  }

  explicit ConvolutionInput(std::span<float, elements> data,
                            std::shared_ptr<memory::Deletable> handle_)
      : store_(std::move(handle_)), data_(data) {}

  ConvolutionInput(const ConvolutionInput& other) = default;
  ConvolutionInput(ConvolutionInput&& other) = default;

  std::span<const float> data() const { return data_; }

  std::span<float> data() { return data_; }

  friend ConvolutionInput Emancipate(const ConvolutionInput& input) {
    // Store has dynamic life span
    if (input.store_ != nullptr) {
      return input;
    }
    // Store owned by inference engine, need to export it.
    ConvolutionInput result;
    auto src = input.data();
    auto dst = result.data();
    std::copy_n(src.data(), elements, dst.data());
    return result;
  }

  float operator()(int channel, int column, int row) const {
    return data_[channel + (column + (row * W)) * C];
  }

  float& operator()(int channel, int column, int row) {
    return data_[channel + (column + (row * W)) * C];
  }

 private:
  std::shared_ptr<memory::Deletable> store_;
  std::span<float, elements> data_;
};

template <typename Input, size_t OutputChannels, size_t KernelHeight,
          size_t KernelWidth, size_t PaddingHeight, size_t PaddingWidth,
          typename Filter>
  requires(OutputChannels % 4 == 0 && KernelHeight > 0 && KernelWidth > 0)
class Conv2dLayer {
 public:
  using input_t = Input;
  using result_t =
      ConvolutionInput<OutputChannels,
                       Input::height + 1 + 2 * PaddingHeight - KernelHeight,
                       Input::width + 1 + 2 * PaddingWidth - KernelWidth>;
  using filtered_result_t =
      std::remove_reference_t<std::invoke_result_t<Filter, result_t&>>;

  constexpr Conv2dLayer() = default;
  template <typename... Args>
  constexpr Conv2dLayer(Args... args)
      : filter_(std::forward<Args...>(args...)) {}

  filtered_result_t operator()(const Input& input, const auto& parameters,
                               auto* ctx) const {
    // Create an explicit span over the scratch area to guarantee the correct
    // extent is used.
    auto* scratch = ctx->GetScratchArea();
    std::span<float, result_t::elements> scratch_span(scratch->data.data(),
                                                      result_t::elements);
    result_t result{scratch_span, nullptr};
    implementation::Conv2d(input.data(), result.data(), parameters,
                           Input::width, kOptions);
    return filter_(result);
  }

 private:
  static constexpr implementation::ConvolutionOptions kOptions = {
      .input_channels = Input::channels,
      .output_channels = OutputChannels,
      .padding_height = PaddingHeight,
      .padding_width = PaddingWidth,
      .kernel_height = KernelHeight,
      .kernel_width = KernelWidth,
  };

  Filter filter_;
};

template <typename Nested = std::identity>
class Flatten {
 public:
  constexpr Flatten() = default;
  constexpr explicit Flatten(Nested nested) : nested_(std::move(nested)) {};

  template <size_t Ch, size_t H, size_t W>
  Vector<float, Ch * H * W> operator()(ConvolutionInput<Ch, H, W> input) const {
    return Vector<float, Ch * H * W>(
        nested_(input).data().template first<Ch * H * W>());
  }

 private:
  Nested nested_;
};

struct ReluFilter {
  template <size_t Ch, size_t H, size_t W>
  ConvolutionInput<Ch, H, W> operator()(
      ConvolutionInput<Ch, H, W> input) const {
    implementation::Relu(input.data());
    return input;
  }
};

template <size_t OutputChannels, size_t KernelHeight, size_t KernelWidth,
          size_t PaddingHeight, size_t PaddingWidth, typename Filter>
class Conv2dLayerDesc {
 public:
  constexpr Conv2dLayerDesc() = default;
  constexpr Conv2dLayerDesc(Filter filter) : filter_(std::move(filter)) {}

  template <typename Layer>
  constexpr auto stack(const Layer& /* layer */) const {
    return Conv2dLayer<typename Layer::output_t, OutputChannels, KernelHeight,
                       KernelWidth, PaddingHeight, PaddingWidth, Filter>(
        filter_);
  }

 private:
  Filter filter_;
};

template <size_t OutputChannels, size_t KernelHeight = 3,
          size_t KernelWidth = KernelHeight, size_t PaddingHeight = 0,
          size_t PaddingWidth = PaddingHeight, typename Filter = std::identity>
static constexpr Layer Conv2d =
    Layer<Conv2dLayerDesc<OutputChannels, KernelHeight, KernelWidth,
                          PaddingHeight, PaddingWidth, Filter>>();

template <size_t OutputChannels, size_t KernelHeight = 3,
          size_t KernelWidth = KernelHeight, size_t PaddingHeight = 0,
          size_t PaddingWidth = PaddingHeight>
constexpr auto Conv2dWithFilter(auto filter) -> Layer<
    Conv2dLayerDesc<OutputChannels, KernelHeight, KernelWidth, PaddingHeight,
                    PaddingWidth, std::remove_cvref_t<decltype(filter)>>> {
  using Conv2dLayer =
      Conv2dLayerDesc<OutputChannels, KernelHeight, KernelWidth, PaddingHeight,
                      PaddingWidth, std::remove_cvref_t<decltype(filter)>>;
  return Layer<Conv2dLayer>(Conv2dLayer(std::move(filter)));
}

template <typename I, size_t OC, size_t KernelHeight, size_t KernelWidth,
          size_t PaddingHeight, size_t PaddingWidth, typename Filter>
auto ParameterProvider(
    const Conv2dLayer<I, OC, KernelHeight, KernelWidth, PaddingHeight,
                      PaddingWidth, Filter>& layer,
    std::span<const float> data, std::shared_ptr<memory::Deletable> ref) {
  return Parameters<OC * KernelHeight * KernelWidth * I::channels>(
      data, std::move(ref));
}

}  // namespace uchen::convolution

template <size_t channels, size_t height, size_t width, size_t OutputChannels,
          size_t KernelHeight, size_t KernelWidth, size_t PaddingHeight,
          size_t PaddingWidth, typename Filter>
struct uchen::LayerTraits<
    uchen::convolution::Conv2dLayer<
        uchen::convolution::ConvolutionInput<channels, height, width>,
        OutputChannels, KernelHeight, KernelWidth, PaddingHeight, PaddingWidth,
        Filter>,
    uchen::convolution::ConvolutionInput<channels, height, width>>
    : public LayerTraitFields<
          typename uchen::convolution::Conv2dLayer<
              uchen::convolution::ConvolutionInput<channels, height, width>,
              OutputChannels, KernelHeight, KernelWidth, PaddingHeight,
              PaddingWidth, Filter>::filtered_result_t,
          KernelHeight * KernelWidth * channels * OutputChannels,
          typename uchen::convolution::Conv2dLayer<
              uchen::convolution::ConvolutionInput<channels, height, width>,
              OutputChannels, KernelHeight, KernelWidth, PaddingHeight,
              PaddingWidth, Filter>::result_t::store_type_t> {};

#endif  // UCHEN_CONVOLUTION_H_