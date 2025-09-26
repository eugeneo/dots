#ifndef UCHEN_TRAINING_RNN_H
#define UCHEN_TRAINING_RNN_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <list>

#include "uchen/layer_traits.h"
#include "uchen/parameters.h"
#include "uchen/rnn.h"
#include "uchen/training/model_gradients.h"

namespace uchen::training {
namespace impl {

template <typename I, typename M, size_t HS>
constexpr size_t Params = LayerTraits<RnnLayer<I, M, HS>, I>::parameter_count;
}

template <typename I, typename M, size_t HS, typename HH>
class RecordingRnnScratchArea final
    : public ::uchen::internal::RnnScratchArea<I, M, HS, HH> {
 private:
  using Base = ::uchen::internal::RnnScratchArea<I, M, HS, HH>;

  class RunIterator {
   private:
    class Run {
     public:
      Run(training::ForwardPassResult<M, typename Base::mm_input_t>* mm,
          training::ForwardPassResult<HH, typename M::output_t>* hh)
          : mm_(mm), hh_(hh) {}

      auto MmGradients(
          const Vector<float, M::output_t::elements>& output_gradients) {
        return mm_->CalculateParameterGradients(output_gradients);
      }

      auto HhGradients(const Vector<float, HS>& output_gradients) {
        return hh_->CalculateParameterGradients(output_gradients);
      }

      bool has_hh() const { return hh_ != nullptr; }

     private:
      training::ForwardPassResult<M, typename Base::mm_input_t>* mm_;
      training::ForwardPassResult<HH, typename M::output_t>* hh_;
    };

   public:
    RunIterator(typename std::list<training::ForwardPassResult<
                    M, typename Base::mm_input_t>>::reverse_iterator mm_iter,
                typename std::list<training::ForwardPassResult<
                    HH, typename M::output_t>>::reverse_iterator hh_iter,
                typename std::list<training::ForwardPassResult<
                    HH, typename M::output_t>>::reverse_iterator hh_end_iter)
        : mm_iter_(mm_iter), hh_iter_(hh_iter), hh_end_iter_(hh_end_iter) {}

    Run operator*() {
      return Run(&(*mm_iter_),
                 hh_iter_ == hh_end_iter_ ? nullptr : &(*hh_iter_));
    }

    bool operator==(const RunIterator& other) const {
      return mm_iter_ == other.mm_iter_ && hh_iter_ == other.hh_iter_;
    }

    bool operator!=(const RunIterator& other) const {
      return !(other == *this);
    }

    RunIterator& operator++() {
      mm_iter_++;
      if (hh_iter_ != hh_end_iter_) {
        hh_iter_++;
      }
      return *this;
    }

   private:
    typename std::list<training::ForwardPassResult<
        M, typename Base::mm_input_t>>::reverse_iterator mm_iter_;
    typename std::list<training::ForwardPassResult<HH, typename M::output_t>>::
        reverse_iterator hh_iter_;
    typename std::list<training::ForwardPassResult<HH, typename M::output_t>>::
        reverse_iterator hh_end_iter_;
  };

 public:
  Vector<typename Base::value_type, HS> hh_run(
      const typename M::output_t& input, const HH& model,
      const ModelParameters<HH>& parameters) override {
    hh_passes_.emplace_back(&model, input, parameters);
    return hh_passes_.back().result();
  }

  typename M::output_t mm_run(const typename Base::mm_input_t& input,
                              const M& model,
                              const ModelParameters<M>& parameters) override {
    mm_passes_.emplace_back(&model, input, parameters);
    return mm_passes_.back().result();
  }

  std::span<typename Base::value_type, HS + M::output_t::elements>
  get_input_store() override {
    input_stores_.emplace_back();
    return input_stores_.back();
  }

  size_t runs() const { return mm_passes_.size(); }

  RunIterator begin() {
    return RunIterator(mm_passes_.rbegin(), hh_passes_.rbegin(),
                       hh_passes_.rend());
  }

  RunIterator end() {
    return RunIterator(mm_passes_.rend(), hh_passes_.rend(), hh_passes_.rend());
  }

 private:
  using input_store_t = std::array<typename Base::mm_input_t::value_type,
                                   Base::mm_input_t::elements>;
  std::list<input_store_t> input_stores_;
  std::list<training::ForwardPassResult<M, typename Base::mm_input_t>>
      mm_passes_;
  std::list<training::ForwardPassResult<HH, typename M::output_t>> hh_passes_;
};

template <typename I, typename M, size_t HS,
          typename HH = typename RnnLayer<I, M, HS>::HH>
Vector<float, 1> ComputeGradients(
    const RnnLayer<I, M, HS>& /* layer */, const I& /* input */,
    Vector<float, M::output_t::elements> output_gradients,
    const Parameters<training::impl::Params<I, M, HS>>& /* parameters */,
    std::span<float, training::impl::Params<I, M, HS>> parameter_gradients,
    ::uchen::internal::RnnScratchArea<I, M, HS, HH>* area) {
  auto* run_record = static_cast<RecordingRnnScratchArea<I, M, HS, HH>*>(area);
  ParameterGradients<M> m_gradients;
  ParameterGradients<HH> hh_gradients;
  for (auto run : *run_record) {
    auto g1 = run.MmGradients(output_gradients);
    m_gradients += g1.second;
    if (run.has_hh()) {
      auto g2 = run.HhGradients(Vector<float, HS>(std::span<float, HS>(
          const_cast<float*>(g1.first.data().data()) + g1.first.size() - HS,
          HS)));
      output_gradients = g2.first;
      hh_gradients += g2.second;
    }
  }
  std::copy(m_gradients.begin(), m_gradients.end(),
            parameter_gradients.begin());
  std::copy(hh_gradients.begin(), hh_gradients.end(),
            parameter_gradients.begin() + m_gradients.size());
  return {0};  // Does not matter
}

template <typename I, typename M, size_t HS, typename NL>
struct ConcreteTypeForGradient<
    ::uchen::internal::RnnScratchArea<I, M, HS, NL>> {
  using type = RecordingRnnScratchArea<I, M, HS, NL>;
};

}  // namespace uchen::training

#endif  // UCHEN_TRAINING_RNN_H