#ifndef UCHEN_LAYER_TRAITS_H
#define UCHEN_LAYER_TRAITS_H

#include <cstddef>

namespace uchen {
namespace details {
struct Empty {};
}  // namespace details

template <typename L, typename I>
struct LayerTraits;

template <typename Output, size_t pc = 0, typename SA = details::Empty,
          bool Skip = false>
struct LayerTraitFields {
  using output_t = Output;
  using scratch_area_t = SA;
  static constexpr size_t parameter_count = pc;
  static constexpr bool needs_parameters = pc > 0;
  static constexpr bool skip = Skip;
};

}  // namespace uchen

#endif  // UCHEN_LAYER_TRAITS_H