#ifndef UCHEN_MEMORY_H
#define UCHEN_MEMORY_H

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <span>
#include <utility>

#include "absl/functional/any_invocable.h"

namespace uchen {
namespace memory {

class Deletable {
 public:
  virtual ~Deletable() = default;
};

template <typename V>
class DeletableAnything : public Deletable {
 public:
  DeletableAnything() = default;
  template <typename... Args>
  DeletableAnything(Args&&... args) : v_(std::forward<Args...>(args...)) {}

  V& operator->() { return v_; }
  V& get() { return v_; }

  static std::unique_ptr<DeletableAnything> NewInstance() {
    return std::make_unique<DeletableAnything>();
  }

 private:
  V v_;
};

template <typename V, size_t C>
class ArrayStore final : public memory::Deletable {
 public:
  ArrayStore() {}
  explicit ArrayStore(V value) {
    for (auto& v : data_) {
      v = value;
    }
  }
  ArrayStore(std::initializer_list<V> init) : data_(init) {}
  explicit ArrayStore(std::span<const V> init) {
    std::copy(std::move_iterator(init.begin()), std::move_iterator(init.end()),
              data_.begin());
  }
  explicit ArrayStore(auto begin, auto end) {
    std::copy(begin, end, data_.begin());
  }

  std::span<V, C> data() { return data_; }
  std::span<const V, C> data() const { return data_; }

  static std::unique_ptr<ArrayStore> NewInstance(V value = V(0)) {
    return std::make_unique<ArrayStore>(value);
  }

  template <typename... Args>
  static std::unique_ptr<ArrayStore> NewInstance(Args&&... args) {
    return std::make_unique<ArrayStore>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<ArrayStore> NewInstance(std::span<const V, C> args) {
    return std::make_unique<ArrayStore>(args);
  }

  constexpr size_t size() const { return C; }

 private:
  std::array<V, C> data_ alignas(16);
};

template <typename ScratchSpace>
class LayerContext {
 public:
  virtual ~LayerContext() = default;
  virtual ScratchSpace* GetScratchArea() = 0;
};

template <typename Model, typename Input>
struct Context {
 private:
  template <size_t I>
  struct LayerFn {
    using type = absl::AnyInvocable<
        typename Model::template Traits<I>::scratch_area_t*()>;
  };

  template <size_t... Ls>
  static auto Init(std::index_sequence<Ls...> indexes) {
    return std::tuple<typename LayerFn<Ls>::type...>();
  }

 protected:
  using vtable_t = decltype(Init(Model::kLayerIndexes));

 public:
  virtual vtable_t& GetLayerArenas() = 0;
};

}  // namespace memory

template <typename Context, typename Target>
struct ConcreteType {
  using type = Target;
};

}  // namespace uchen

#endif  // UCHEN_MEMORY_H