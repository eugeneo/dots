#ifndef WASM_SRC_GAME_H
#define WASM_SRC_GAME_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <ostream>
#include <set>
#include <span>
#include <unordered_set>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/strings/substitute.h"

namespace uchen::demo {

class Game {
 public:
  static constexpr size_t kBufferSize = 64 * 64;

  class Polygon {
   public:
    Polygon(int x, int y, int width, std::string_view elements, size_t captures,
            int player);
    explicit Polygon(int x, int y, int width, std::vector<bool> data,
                     size_t captures, int player)
        : x_(x),
          y_(y),
          width_(width),
          data_(std::move(data)),
          captures_(captures),
          player_(player) {}

    bool operator==(const Polygon& other) const {
      return x_ == other.x_ && y_ == other.y_ && data_ == other.data_;
    }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Polygon& polygon) {
      sink.Append(absl::Substitute("($0, $1) $2", polygon.x_, polygon.y_,
                                   polygon.StrData()));
    }

    template <typename H>
    friend H AbslHashValue(H h, const Polygon& m) {
      return H::combine(std::move(h), m.data_, m.x_, m.y_, m.width_);
    }

   private:
    std::string StrData() const;

    int x_, y_, width_;
    std::vector<bool> data_;
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