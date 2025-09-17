#ifndef WASM_SRC_GAME_H
#define WASM_SRC_GAME_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>

#include "absl/container/inlined_vector.h"

namespace uchen::demo {

class Game {
public:
  Game(size_t height, size_t width);
  std::span<const uint8_t> field() const { return field_; }
  void GameTurn(size_t index, uint8_t player_id);

  size_t player_score(uint8_t player_id) const {
    auto it = player_scores_.find(player_id);
    return it == player_scores_.end() ? 0 : it->second;
  }
  size_t width() const { return width_; }
  int player_at(int x, int y) const { return field_[y * width_ + x] & 0x5F; }

private:
  bool is_polygon(size_t x, size_t y) const {
    return (field_[y * width_ + x] & 0x40) != 0;
  }

  void set_dot(int x, int y, uint8_t player_id, bool is_polygon) {
    field_[y * width_ + x] = player_id | (is_polygon * 0x40);
  }

  size_t width_;
  absl::InlinedVector<uint8_t, 64 * 64> field_;
  std::map<uint8_t, size_t> player_scores_;
};

}; // namespace uchen::demo

#endif // WASM_SRC_GAME_H