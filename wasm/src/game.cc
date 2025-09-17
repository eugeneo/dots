#include "src/game.h"
#include <cstddef>

namespace uchen::demo {

Game::Game(size_t height, size_t width)
    : width_(width), field_(height * width, 0) {}

void Game::GameTurn(size_t index, uint8_t player_id) {
  int x = index % width_;
  int y = index / width_;
  set_dot(x, y, player_id, false);
  // Detect the polygon and fill it. Update player score - every enemy dot gives
  // 10 points, empty dot gives 1 point.
}

} // namespace uchen::demo