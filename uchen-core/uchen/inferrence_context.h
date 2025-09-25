#ifndef UCHEN_INFERRENCE_CONTEXT_H
#define UCHEN_INFERRENCE_CONTEXT_H

#include <cstddef>
#include <tuple>
#include <utility>
#include <variant>

#include "uchen/memory.h"

namespace uchen {

template <typename M, typename I>
class ContextForInfer final : public memory::Context<M, I> {
 public:
  ContextForInfer() : vtable_(MakeVtable(M::kLayerIndexes)) {}

  typename memory::Context<M, I>::vtable_t& GetLayerArenas() { return vtable_; }

 private:
  template <size_t Ind>
  auto GetterForLayer() {
    static_assert(Ind < M::kLayers, "Index out of range");
    using S = typename M::template Traits<Ind>::scratch_area_t;
    return absl::AnyInvocable<S*()>([ctx = this]() mutable {
      S* ptr = nullptr;
      if constexpr ((Ind & 1) == 0) {
        ptr = std::get_if<Ind / 2>(&ctx->evens_);
        if (ptr == nullptr) {
          ctx->evens_.template emplace<Ind / 2>();
          ptr = &std::get<Ind / 2>(ctx->evens_);
        }
      } else {
        ptr = std::get_if<Ind / 2>(&ctx->odds_);
        if (ptr == nullptr) {
          ctx->odds_.template emplace<Ind / 2>();
          ptr = &std::get<Ind / 2>(ctx->odds_);
        }
      }
      return ptr;
    });
  }

  template <size_t... L>
  auto MakeVtable(std::index_sequence<L...> seq) {
    return std::make_tuple(GetterForLayer<L>()...);
  }

  struct Areas {
    template <size_t... Ls>
    static auto InitTuple(std::index_sequence<Ls...> /* seq */) {
      return std::tuple<typename M::template Traits<Ls>::scratch_area_t...>();
    }

    using type_t = decltype(InitTuple(M::kLayerIndexes));
  };

  template <size_t II>
  using CT = typename ConcreteType<
      ContextForInfer, typename M::template Traits<II>::scratch_area_t>::type;

  struct Odds {
    template <size_t... Is>
    static auto InitVariant(std::index_sequence<Is...> /* seq */) {
      if constexpr (sizeof...(Is) == 0) {
        return std::variant<int>();
      } else {
        return std::variant<CT<1 + Is * 2>...>();
      }
    }

    using type_t =
        decltype(InitVariant(std::make_index_sequence<M::kLayers / 2>()));
  };

  struct Evens {
    template <size_t... Is>
    static auto InitVariant(std::index_sequence<Is...> /* seq */) {
      if constexpr (sizeof...(Is) == 0) {
        return std::variant<int>();
      } else {
        return std::variant<CT<Is * 2>...>();
      }
    }

    using type_t =
        decltype(InitVariant(std::make_index_sequence<(M::kLayers + 1) / 2>()));
  };

  typename Evens::type_t evens_;
  typename Odds::type_t odds_;
  typename memory::Context<M, I>::vtable_t vtable_;
};

}  // namespace uchen

#endif  // UCHEN_INFERRENCE_CONTEXT_H