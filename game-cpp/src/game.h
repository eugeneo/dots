#ifndef WASM_SRC_GAME_H
#define WASM_SRC_GAME_H

#include <cstddef>
#include <cstdint>
#include <map>
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

  class PlayerOverlay {
   public:
    PlayerOverlay(size_t w, size_t h) : width_(w), data_(h * w, false) {}

    int width() const { return width_; }
    int height() const { return data_.size() / width_; }
    void set_dot(size_t index, bool v) { data_[index] = v; }
    bool get_dot(size_t index) const { return data_[index]; }
    void set_captured(size_t index) { captured_.emplace(index); }
    size_t captured() const { return captured_.size(); }

    friend bool operator==(const PlayerOverlay& a, const PlayerOverlay& b) {
      return a.width_ == b.width_ && a.data_ == b.data_;
    }

    friend void AbslStringify(auto& sink, const PlayerOverlay& polygon) {
      std::string data =
          absl::StrJoin(polygon.data_, "", [](std::string* s, bool data) {
            absl::StrAppend(s, data ? "x" : ".");
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
    std::vector<bool> data_;
    std::unordered_set<size_t> captured_;
  };

  Game(int height, int width);

  void PlaceDot(size_t index, uint8_t player_id);

  std::span<const uint8_t> field() const { return field_; }
  size_t player_score(uint8_t player_id) const {
    auto it = player_scores_.find(player_id);
    return it == player_scores_.end() ? 0 : it->second;
  }
  size_t width() const { return width_; }

  // Exposed for tests
  absl::InlinedVector<size_t, kBufferSize> PathBetween(
      size_t start, size_t end,
      const std::set<std::pair<size_t, size_t>>& ignored_transitions = {})
      const;

  void FillPolygons(int x, int y);

  PlayerOverlay& player_overlay(int player_id) ABSL_ATTRIBUTE_LIFETIME_BOUND {
    size_t pip = player_id - 1;
    while (overlays_.size() <= pip) {
      overlays_.emplace_back(width_, field_.size() / width_);
    }
    return overlays_[pip];
  }

 private:
  int player_at(size_t index) const { return field_[index]; }

  void set_dot(int x, int y, uint8_t player_id) {
    field_[y * width_ + x] = player_id;
  }

  void FillPath(std::span<const size_t> path, int player_id);

  int width_;
  absl::InlinedVector<uint8_t, kBufferSize> field_;
  std::map<uint8_t, size_t> player_scores_;
  std::vector<PlayerOverlay> overlays_;
};

};  // namespace uchen::demo

#endif  // WASM_SRC_GAME_H