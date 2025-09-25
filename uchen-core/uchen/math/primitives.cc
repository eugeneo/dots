#include "uchen/math/primitives.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "absl/log/check.h"  // IWYU pragma: keep
#include "absl/log/log.h"    // IWYU pragma: keep

#undef HWY_SHARED_DEFINE
#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"
#include "hwy/print-inl.h"

namespace uchen::math {

namespace hn = ::hwy::HWY_NAMESPACE;
HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {

namespace {

template <typename D>
hn::VFromD<D> ColumnExpSum(D d, const float* HWY_RESTRICT inp,
                           float* HWY_RESTRICT out, hn::VFromD<D> max,
                           size_t split, size_t end) {
  using V = hn::VFromD<D>;
  V accum = Zero(d);
  hn::Transform1(d, out, split, inp, [max, &accum](D d, V, V in) {
    auto exp = Exp(d, Sub(in, max));
    accum = Add(accum, exp);
    return exp;
  });
  V remaining = Zero(d);
  if (split <= end) {
    auto m = FirstN(d, end - split);
    remaining =
        IfThenElseZero(m, Exp(d, Sub(LoadN(d, inp + split, end - split), max)));
    StoreN(remaining, d, out + split, end & (hn::Lanes(d) - 1));
  }
  return SumOfLanes(d, Add(accum, remaining));
}

struct FullVector {
  template <typename D>
  hn::VFromD<D> Load(D d, const float* HWY_RESTRICT in) const {
    return LoadU(d, in);
  }

  template <typename D>
  void Store(hn::VFromD<D> v, D d, float* HWY_RESTRICT out) const {
    StoreU(v, d, out);
  }
};

class PartialVector {
 public:
  explicit PartialVector(size_t lanes) : lanes(lanes) {}

  template <typename D>
  hn::VFromD<D> Load(D d, const float* HWY_RESTRICT in) const {
    return LoadN(d, in, lanes);
  }

  template <typename D>
  void Store(hn::VFromD<D> v, D d, float* HWY_RESTRICT out) const {
    StoreN(v, d, out, lanes);
  }

 private:
  size_t lanes;
};

template <typename D>
hn::VFromD<D> ParallelMax(D d, const float* HWY_RESTRICT in, size_t count,
                          size_t stride, const auto& ld) {
  using V = hn::VFromD<D>;
  V max = Set(d, -std::numeric_limits<float>::infinity());
  for (size_t r = 0; r < count; ++r) {
    max = Max(max, ld.Load(d, in));
    in += stride;
  }
  return max;
}

template <typename D>
hn::VFromD<D> ParallelExpSum(D d, const float* HWY_RESTRICT in,
                             float* HWY_RESTRICT out, hn::VFromD<D> max,
                             size_t count, size_t stride, const auto& ld) {
  using V = hn::VFromD<D>;
  V sum = Zero(d);
  for (size_t r = 0; r < count; ++r) {
    V exp = Exp(d, Sub(ld.Load(d, in), max));
    sum = Add(sum, exp);
    ld.Store(exp, d, out);
    in += stride;
    out += stride;
  }
  return sum;
}

template <typename D>
void ParallelDiv(D d, float* HWY_RESTRICT inout, hn::VFromD<D> sum,
                 size_t count, size_t stride, const auto& ld) {
  for (size_t r = 0; r < count; ++r) {
    ld.Store(Div(ld.Load(d, inout), sum), d, inout);
    inout += stride;
  }
}

template <typename D>
void ParallelSoftmax(D d, const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                     size_t count, size_t stride, const auto& ld) {
  using V = hn::VFromD<D>;
  V max = ParallelMax(d, in, count, stride, ld);
  V sum = ParallelExpSum(d, in, out, max, count, stride, ld);
  ParallelDiv(d, out, sum, count, stride, ld);
}

}  // namespace

HWY_ATTR void ColumnsByRow(std::span<const float> a, std::span<const float> b,
                           std::span<float> out) {
  DCHECK_EQ(a.size() % b.size(), 0u);
  DCHECK_EQ(out.size(), a.size() / b.size());
  using D = hn::ScalableTag<float>;
  constexpr D d;
  using V = hn::VFromD<D>;
  std::fill(out.begin(), out.end(), 0.f);
  for (size_t i = 0; i < b.size(); ++i) {
    hn::Transform1(d, out.data(), out.size(), a.subspan(i * out.size()).data(),
                   [multiplier = Set(d, b[i])](D, V inout, V in) {
                     return hn::MulAdd(in, multiplier, inout);
                   });
  }
}

HWY_ATTR void CWSoftmax(std::span<const float> in, std::span<float> out,
                        uint32_t rows) {
  using D = hn::ScalableTag<float>;
  constexpr D d;
  using V = hn::VFromD<D>;
  const size_t end = rows & ~static_cast<size_t>(hn::Lanes(d) - 1);
  const float* inp = in.data();
  float* o = out.data();
  for (size_t c = 0; c < in.size(); c += rows) {
    V max = Set(d, -std::numeric_limits<float>::infinity());
    hn::Foreach(d, inp, rows, max, [&max](D d, V v) { max = Max(max, v); });
    max = hn::MaxOfLanes(d, max);
    V divisor = ColumnExpSum(d, inp, o, max, end, rows);
    hn::Transform(d, o, rows,
                  [divisor](D, hn::VFromD<D> v) { return Div(v, divisor); });
    inp += rows;
    o += rows;
  }
}

HWY_ATTR void RWSoftmax(std::span<const float> in, std::span<float> out,
                        uint32_t cols) {
  using D = hn::ScalableTag<float>;
  constexpr D d;
  const size_t rows = in.size() / cols;
  size_t full_columns = cols & ~static_cast<size_t>(hn::Lanes(d) - 1);
  FullVector full;
  for (size_t c = 0; c < full_columns; c += hn::Lanes(d)) {
    ParallelSoftmax(d, in.data() + c, out.data() + c, rows, cols, full);
  }
  if (full_columns != cols) {
    ParallelSoftmax(d, in.data() + full_columns, out.data() + full_columns,
                    rows, cols, PartialVector(cols - full_columns));
  }
}

HWY_ATTR uint16_t GetLanes() { return hn::Lanes(hn::ScalableTag<float>{}); }

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

float DotProduct(std::span<const float> a, std::span<const float> b) {
  CHECK_EQ(a.size(), b.size());
  if (a.size() < 4) {  // Too small for HWY
    float sum = 0;
    for (uint32_t i = 0; i < a.size(); ++i) {
      sum += a[i] * b[i];
    }
    return sum;
  }
  const hn::ScalableTag<float> d;
  return hn::Dot::Compute<hn::Dot::Assumptions::kAtLeastOneVector>(
      d, a.data(), b.data(), a.size());
}

void MatrixByVector(std::span<const float> a, std::span<const float> b,
                    std::span<float> out) {
  HWY_STATIC_DISPATCH(ColumnsByRow)(a, b, out);
}

void ColumnWiseSoftmax(std::span<const float> in, std::span<float> out,
                       uint32_t rows) {
  HWY_STATIC_DISPATCH(CWSoftmax)(in, out, rows);
}

void RowWiseSoftmax(std::span<const float> in, std::span<float> out,
                    uint32_t cols) {
  HWY_STATIC_DISPATCH(RWSoftmax)(in, out, cols);
}

uint16_t GetLanesForTest() { return HWY_STATIC_DISPATCH(GetLanes)(); }

}  // namespace uchen::math
