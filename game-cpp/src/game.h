#ifndef WASM_SRC_GAME_H
#define WASM_SRC_GAME_H

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <set>
#include <span>
#include <unordered_set>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/substitute.h"

namespace uchen::demo {

class Game {
 public:
  static constexpr size_t kBufferSize = 64 * 64;

  class Grid {
   public:
    Grid(int x, int y, size_t w, size_t h)
        : x_(x), y_(y), width_(w), height_(h), data_(w * h, State::kUnknown) {}

    int width() const { return width_; }
    int height() const { return height_; }
    int x() const { return x_; }
    int y() const { return y_; }
    void mark_inside(size_t index) { data_[index] = State::kInside; }
    void mark_outside(size_t index) { data_[index] = State::kOutside; }
    bool is_unknown(size_t index) const {
      return data_[index] == State::kUnknown;
    }
    bool is_outside(size_t index) const {
      return data_[index] == State::kOutside;
    }

   private:
    enum class State { kUnknown, kInside, kOutside };

    int x_, y_;
    size_t width_, height_;
    absl::InlinedVector<State, Game::kBufferSize> data_;
  };

  class PlayerOverlay {
   public:
    PlayerOverlay(size_t w, size_t h, int player_id)
        : width_(w), data_(h * w, 0), player_id_(player_id) {}

    int width() const { return width_; }
    int height() const { return data_.size() / width_; }
    uint16_t get_dot(size_t index) const { return data_[index]; }
    size_t captured_count() const { return captured_.size(); }
    bool captured(size_t index) const { return captured_.contains(index); }
    void set_dot(size_t index, uint16_t v) { data_[index] = v; }
    void set_captured(size_t index) { captured_.emplace(index); }

    void MarkRegion(const Grid& grid, const Game& game);

    friend bool operator==(const PlayerOverlay& a, const PlayerOverlay& b) {
      return a.width_ == b.width_ && a.data_ == b.data_;
    }

    friend void AbslStringify(auto& sink, const PlayerOverlay& polygon) {
      std::string data =
          absl::StrJoin(polygon.data_, "", [](std::string* s, uint16_t data) {
            absl::StrAppend(
                s, data > 10 ? "x" : (data == 0 ? "." : absl::StrCat(data)));
          });
      sink.Append(absl::Substitute(
          "($0x$1) |$2|", polygon.width_, polygon.data_.size() / polygon.width_,
          absl::StrJoin(absl::StrSplit(data, absl::ByLength(polygon.width_)),
                        "|")));
    }

    friend std::ostream& operator<<(std::ostream& os,
                                    const PlayerOverlay& overlay) {
      os << absl::StrCat(overlay);
      return os;
    }

   private:
    size_t width_;
    std::vector<uint16_t> data_;
    std::unordered_set<size_t> captured_;
    uint16_t next_region_id_ = 1;
    int player_id_;
  };

  struct Polygon {
    enum class Direction : uint8_t {
      kN = 0,
      kNe = 1,
      kE = 2,
      kSe = 3,
      kS = 4,
      kSw = 5,
      kW = 6,
      kNw = 7,
    };

    static constexpr std::array<std::pair<int, int>, 8> kDirections = {
        {{0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}}};
    static constexpr std::array<std::string_view, 8> kDirectionNames = {
        "N", "NE", "E", "SE", "S", "SW", "W", "NW"};

    size_t x, y;
    std::vector<Direction> outline;
    int player;

    friend bool operator==(const Polygon& a, const Polygon& b) {
      return a.x == b.x && a.y == b.y && a.outline == b.outline &&
             a.player == b.player;
    }

    friend void AbslStringify(auto& sink, const Polygon& polygon) {
      std::string data =
          absl::StrJoin(polygon.outline, "-", [](std::string* s, Direction d) {
            absl::StrAppend(s, kDirectionNames[static_cast<size_t>(d)]);
          });
      sink.Append(absl::Substitute("($0,$1)p$2 [$3]", polygon.x, polygon.y,
                                   polygon.player, data));
    }

    friend std::ostream& operator<<(std::ostream& os, const Polygon& polygon) {
      os << absl::StrCat(polygon);
      return os;
    }
  };

  Game(int height, int width);

  /* Returns true if regions were updated */
  bool PlaceDot(size_t index, uint8_t player_id);

  std::span<const uint8_t> field() const { return field_; }
  size_t player_score(uint8_t player_id) const {
    if (overlays_.size() < player_id || player_id == 0) {
      return 0;
    }
    return overlays_[player_id - 1].captured_count();
  }
  size_t width() const { return width_; }

  // Exposed for tests
  absl::InlinedVector<size_t, kBufferSize> PathBetween(
      size_t start, size_t end,
      const std::set<std::pair<size_t, size_t>>& ignored_transitions = {})
      const;

  bool FillPolygons(int x, int y);

  PlayerOverlay& player_overlay(int player_id) ABSL_ATTRIBUTE_LIFETIME_BOUND {
    size_t pip = player_id - 1;
    while (overlays_.size() <= pip) {
      overlays_.emplace_back(width_, field_.size() / width_,
                             overlays_.size() + 1);
    }
    return overlays_[pip];
  }

  std::span<const PlayerOverlay> player_overlays() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return overlays_;
  }

  std::span<const Polygon> polygons() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return polygons_;
  }

 private:
  int player_at(size_t index) const { return field_[index]; }

  void set_dot(int x, int y, uint8_t player_id) {
    field_[y * width_ + x] = player_id;
  }

  void FillPath(std::span<const size_t> path, int player_id);

  bool Captured(size_t index) const;

  int width_;
  absl::InlinedVector<uint8_t, kBufferSize> field_;
  std::vector<PlayerOverlay> overlays_;
  std::vector<Polygon> polygons_;
};

};  // namespace uchen::demo

#endif  // WASM_SRC_GAME_H