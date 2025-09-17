#include <cstdint>
#include <string>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "src/game.h"

namespace test {

int add(int a, int b);
std::string simd();

} // namespace test

class GameWrapper {
public:
  GameWrapper(int h, int w) : game_(h, w) {
    field_ = emscripten::val(emscripten::typed_memory_view(
        game_.field().size(), game_.field().data()));
  }

  emscripten::val GetField() const { return field_; }

  void GameTurn(size_t index, uint8_t player_id) {
    game_.GameTurn(index, player_id);
  }

  size_t player_score(uint8_t player_id) const {
    return game_.player_score(player_id);
  }

private:
  uchen::demo::Game game_;
  emscripten::val field_;
};

EMSCRIPTEN_BINDINGS(dots_logic) {
  emscripten::class_<GameWrapper>("Game")
      .constructor<int, int>()
      .function("field", &GameWrapper::GetField)
      .function("doTurn", &GameWrapper::GameTurn)
      .function("playerScore", &GameWrapper::player_score);
}

int main() { return 0; }