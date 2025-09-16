#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include "absl/strings/str_join.h"
#include "hwy/highway.h"

namespace test {

namespace hn = ::hwy::HWY_NAMESPACE;
HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {

// void VectorAdd(const float *a, const float *b, float *c, size_t N) {
//   hn::ScalableTag<float> d;
//   hn::
// }

// 2000000000000000
std::string simd() {
  std::array arr = {HWY_SCALAR, HWY_EMU128, HWY_WASM, HWY_WASM_EMU256};
  std::stringstream ss;
  hn::ScalableTag<float> d;
  ss << typeid(d).name() << " "
     << absl::StrJoin(arr, " - ", [](std::string *out, auto t) {
          absl::StrAppend(out, t == HWY_TARGET);
        });
  return ss.str();
}

} // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

int add(int a, int b) { return a + b; }

std::string simd() { return HWY_DYNAMIC_DISPATCH(simd)(); }

} // namespace test