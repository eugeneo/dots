#ifndef WASM_SRC_GAME_H
#define WASM_SRC_GAME_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <map>
#include <ostream>
#include <set>
#include <span>
#include <unordered_set>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_join.h"

namespace uchen::demo {

class Game {
 public:
  static constexpr size_t kBufferSize = 64 * 64;

  class Polygon {
   public:
    Polygon(std::initializer_list<size_t> elements, size_t captures, int player)
        : dots_(elements), captures_(captures), player_(player) {}
    explicit Polygon(std::span<const size_t> dots, size_t captures,
                     int player) {
      std::copy(dots.begin(), dots.end(), std::inserter(dots_, dots_.end()));
    }

    bool operator==(const Polygon& other) const { return dots_ == other.dots_; }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Polygon& polygon) {
      sink.Append(absl::StrJoin(polygon.dots_, "-"));
    }

    template <typename H>
    friend H AbslHashValue(H h, const Polygon& m) {
      return H::combine(std::move(h), m.dots_);
    }

   private:
    std::set<size_t> dots_;
    size_t captures_;
    int player_;
  };

  Game(int height, int width);

  void PlaceDot(size_t index, uint8_t player_id);

  std::span<const uint8_t> field() const { return field_; }
  size_t player_score(uint8_t player_id) const {
    auto it = player_scores_.find(player_id);
    return it == player_scores_.end() ? 0 : it->second;
  }
  std::span<const Polygon> polygons() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return polygons_;
  }
  size_t width() const { return width_; }

  // Exposed for tests
  absl::InlinedVector<size_t, kBufferSize> PathBetween(
      size_t start, size_t end,
      const std::set<std::pair<size_t, size_t>>& ignored_transitions = {})
      const;

  std::unordered_set<Polygon> DetectPolygons(int x, int y) const;

 private:
  int player_at(size_t index) const { return field_[index]; }

  void set_dot(int x, int y, uint8_t player_id) {
    field_[y * width_ + x] = player_id;
  }

  std::optional<Game::Polygon> PolygonFromPath(std::span<const size_t> path,
                                               int player_id) const;

  int width_;
  absl::InlinedVector<uint8_t, kBufferSize> field_;
  std::map<uint8_t, size_t> player_scores_;
  std::vector<Polygon> polygons_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const Game::Polygon& polygon) {
  os << absl::StrCat(polygon);
  return os;
}

};  // namespace uchen::demo

template <>
struct std::hash<uchen::demo::Game::Polygon> {
  size_t operator()(const uchen::demo::Game::Polygon& polygon) const {
    return absl::Hash<uchen::demo::Game::Polygon>()(polygon);
  }
};

#endif  // WASM_SRC_GAME_H