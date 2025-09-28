#ifndef UCHEN_TRAINING_TRAINING_H
#define UCHEN_TRAINING_TRAINING_H

#include <algorithm>
#include <array>
#include <barrier>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "parameter_gradients.h"
#include "uchen/parameters.h"
#include "uchen/training/loss.h"
#include "uchen/training/model_gradients.h"

namespace uchen::training {

template <typename V>
class Store {
 public:
  virtual ~Store() = default;
  virtual size_t size() const = 0;
  virtual const V& operator[](size_t index) const = 0;
};

template <typename V>
class InlineStore final : public Store<V> {
 public:
  InlineStore(std::initializer_list<V> data) : data_(data) {}
  template <typename I1, typename I2>
  InlineStore(I1 begin, I2 end) : data_(begin, end) {}

  size_t size() const override { return data_.size(); }
  const V& operator[](size_t index) const override { return data_[index]; }

 private:
  std::vector<V> data_;
};

template <typename V>
class Projection final : public Store<V> {
 public:
  Projection(std::shared_ptr<Store<V>> store, size_t from, size_t to)
      : store_(std::move(store)), from_(from), to_(to) {
    DCHECK_GE(to_, from_);
  }

  size_t size() const override { return to_ - from_; }

  const V& operator[](size_t index) const override {
    DCHECK_LT(from_ + index, to_);
    return (*store_)[from_ + index];
  }

 private:
  std::shared_ptr<Store<V>> store_;
  size_t from_;
  size_t to_;
};

template <typename V>
class ShuffledStore final : public Store<V> {
 public:
  explicit ShuffledStore(std::shared_ptr<Store<V>> store)
      : store_(std::move(store)) {
    indexes_.reserve(store_->size());
    for (size_t i = 0; i < store_->size(); ++i) {
      indexes_.emplace_back(i);
    }
    std::shuffle(indexes_.begin(), indexes_.end(),
                 std::default_random_engine());
  }

  size_t size() const override { return indexes_.size(); }

  const V& operator[](size_t index) const override {
    return (*store_)[indexes_[index]];
  }

 private:
  std::shared_ptr<Store<V>> store_;
  std::vector<size_t> indexes_;
};

template <typename V>
class ReferenceStore final : public Store<V> {
 public:
  explicit ReferenceStore(std::span<const V> data,
                          std::shared_ptr<Store<V>> store)
      : data_(data), store_(std::move(store)) {}
  size_t size() const override { return data_.size(); }
  const V& operator[](size_t index) const override { return data_[index]; }

 private:
  std::span<const V> data_;
  std::shared_ptr<Store<V>> store_;
};

template <typename Input, typename Expected>
class TrainingData {
 public:
  using value_type = std::pair<Input, Expected>;

  class Iterator {
   public:
    Iterator(std::shared_ptr<Store<value_type>> store, size_t index)
        : store_(store), index_(index) {}

    bool operator==(const Iterator& other) const {
      return store_ == other.store_ && index_ == other.index_;
    }

    bool operator!=(const Iterator& other) const { return !(*this == other); }

    Iterator& operator++(int /* unused */) {
      ++index_;
      return *this;
    }

    Iterator operator++() {
      Iterator it(store_, index_);
      ++index_;
      return it;
    }

    const value_type& operator*() const { return (*store_)[index_]; }

   private:
    std::shared_ptr<Store<value_type>> store_;
    size_t index_;
  };

  TrainingData(std::initializer_list<value_type> data)
      : store_(std::make_shared<InlineStore<value_type>>(std::move(data))) {}

  TrainingData(auto begin_it, auto end_it) {
    store_ = std::make_shared<InlineStore<value_type>>(begin_it, end_it);
  }

  explicit TrainingData(std::shared_ptr<Store<value_type>> store)
      : store_(std::move(store)) {}

  bool empty() const { return store_->size() == 0; }

  auto begin() const { return Iterator(store_, 0); }

  auto end() const { return Iterator(store_, store_->size()); }

  size_t size() const { return store_->size(); }

  const value_type& operator[](size_t i) const { return (*store_)[i]; }

  std::pair<TrainingData, TrainingData> Split(float ratio) const {
    DCHECK_LE(ratio, 1);
    size_t arr1 = ratio * size();
    return std::make_pair(
        TrainingData(std::make_shared<Projection<value_type>>(store_, 0, arr1)),
        TrainingData(
            std::make_shared<Projection<value_type>>(store_, arr1, size())));
  }

  TrainingData Shuffle() const {
    return TrainingData(std::make_shared<ShuffledStore<value_type>>(store_));
  }

  std::vector<TrainingData> BatchWithSize(size_t batch_size) const {
    std::vector<TrainingData> result;
    result.reserve((size() + batch_size / 2) / batch_size);
    size_t i = 0;
    for (; (i + batch_size) < size(); i += batch_size) {
      result.emplace_back(TrainingData(
          std::make_shared<Projection<value_type>>(store_, i, i + batch_size)));
    }
    if (i < size()) {
      result.emplace_back(TrainingData(
          std::make_shared<Projection<value_type>>(store_, i, size())));
    }
    return result;
  }

  template <size_t C>
  static TrainingData FromArray(
      std::array<std::pair<Input, Expected>, C>&& data) {
    return TrainingData(std::make_move_iterator(data.begin()),
                        std::make_move_iterator(data.end()));
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TrainingData& p) {
    std::vector<std::string> vals;
    for (size_t i = 0; i < p.size() && i < 3; ++i) {
      vals.emplace_back(absl::StrFormat("(%v, %v)", p[i].first, p[i].second));
    }
    if (p.size() > 3) {
      vals.emplace_back("...");
    }
    absl::Format(&sink, "[%v samples]{%s}", p.size(),
                 absl::StrJoin(vals, ", "));
  }

 private:
  // In case the data is stored inline
  std::shared_ptr<Store<value_type>> store_;
};

template <typename M,
          typename L = typename DefaultLoss<typename M::output_t>::type>
class Training {
 public:
  Training(const M* model, const ModelParameters<M>& parameters,
           L loss_fn = L())
      : model_(model), parameters_(parameters), loss_fn_(loss_fn) {}

  template <typename I>
  double Loss(const TrainingData<I, typename L::value_type>& data_set) const {
    if (data_set.empty()) {
      return 0;
    }
    std::atomic<double> loss = 0;
    std::vector batches =
        data_set.BatchWithSize(data_set.size() / tasks(data_set.size()));
    std::vector<std::jthread> threads;
    for (const auto& batch : batches) {
      threads.emplace_back([&loss, batch, this]() {
        double l = 0;
        for (const auto& [input, y_hat] : batch) {
          const auto y = (*model_)(input, parameters_);
          l += loss_fn_.Loss(y, y_hat);
        }
        loss += l;
      });
    }
    threads.clear();  // All finished
    return loss / data_set.size();
  }

  template <typename I>
  Training Generation(const TrainingData<I, typename L::value_type>& data_set,
                      float learning_rate, double* out_loss = nullptr) const {
    if (data_set.empty()) {
      if (out_loss != nullptr) {
        *out_loss = 0;
      }
      return *this;
    }

    std::vector batches =
        data_set.BatchWithSize(data_set.size() / tasks(data_set.size()));
    std::vector<GradientWorker> workers;
    std::vector<std::jthread> runners;
    std::vector<std::pair<ParameterGradients<M>, float>> gradients_losses(
        batches.size());
    std::barrier sync{static_cast<unsigned int>(batches.size() + 1), []() {}};
    for (size_t i = 0; i < batches.size(); ++i) {
      workers.emplace_back(model_, batches[i], parameters_, loss_fn_);
    }
    for (auto& worker : workers) {
      runners.emplace_back([&]() { worker(); });
    }
    // Joins the threads!
    runners.clear();
    auto& [gradients, loss] = gradients_losses.front();
    for (size_t i = 1; i < gradients_losses.size(); ++i) {
      loss += gradients_losses[i].second;
      gradients += gradients_losses[i].first;
    }
    if (out_loss != nullptr) {
      for (const auto& worker : workers) {
        *out_loss = worker.loss() / data_set.size();
      }
    }
    ParameterGradients grads{model_};
    for (const auto& worker : workers) {
      grads += worker.gradients();
    }
    return Training(model_,
                    parameters_ - grads * (learning_rate / data_set.size()));
  }

  ModelParameters<M> parameters() const { return parameters_; }

 private:
  class GradientWorker {
   public:
    explicit GradientWorker(
        const M* model,
        TrainingData<typename M::input_t, typename L::value_type> batch,
        ModelParameters<M> parameters, L loss_fn_)
        : model_(model),
          batch_(std::move(batch)),
          loss_fn_(std::move(loss_fn_)),
          parameters_(parameters) {}

    void operator()() {
      for (const auto& [input, y_hat] : batch_) {
        ForwardPassResult fpr(model_, input, parameters_);
        auto per_run_gradients = fpr.CalculateParameterGradients(
                                        loss_fn_.Gradient(fpr.result(), y_hat))
                                     .second;
        float per_run_loss = loss_fn_.Loss(fpr.result(), y_hat);
        gradients_ += per_run_gradients;
        loss_ += per_run_loss;
      }
    }

    const ParameterGradients<M>& gradients() const { return gradients_; }

    float loss() const { return loss_; }

   private:
    const M* model_;
    TrainingData<typename M::input_t, typename L::value_type> batch_;
    ParameterGradients<M> gradients_;
    float loss_;
    L loss_fn_;
    ModelParameters<M> parameters_;
  };

  uint32_t tasks(size_t data_samples) const {
    return std::max(std::min(std::thread::hardware_concurrency(),
                             static_cast<unsigned int>(data_samples / 5)),
                    1u);
  }

  const M* model_;
  ModelParameters<M> parameters_;
  L loss_fn_;
};

}  // namespace uchen::training

namespace std {
template <typename I, typename Y>
std::ostream& operator<<(std::ostream& stream,
                         const uchen::training::TrainingData<I, Y>& data) {
  stream << absl::StrCat(data);
  return stream;
}

}  // namespace std

#endif  // UCHEN_TRAINING_TRAINING_H