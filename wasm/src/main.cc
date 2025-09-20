#include <cstdint>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "absl/log/log.h"
#include "src/game.h"

using uchen::demo::Game;

class GameWrapper {
public:
  GameWrapper(int h, int w) : game_(h, w), regions_(emscripten::val::array()) {
    field_ = emscripten::val(emscripten::typed_memory_view(
        game_.field().size(), game_.field().data()));
  }

  emscripten::val GetField() const { return field_; }

  void GameTurn(size_t index, uint8_t player_id) {
    game_.PlaceDot(index, player_id);
  }

  size_t player_score(uint8_t player_id) const {
    return game_.player_score(player_id);
  }

  std::vector<std::string> get_regions() const { return {"a", "b"}; }

private:
  Game game_;
  emscripten::val field_;
  emscripten::val regions_;
};

EMSCRIPTEN_BINDINGS(dots_logic) {
  emscripten::register_vector<std::string>("StrVector");
  emscripten::class_<GameWrapper>("Game")
      .constructor<int, int>()
      .function("field", &GameWrapper::GetField)
      .function("doTurn", &GameWrapper::GameTurn)
      .function("playerScore", &GameWrapper::player_score)
      .function("regions", &GameWrapper::get_regions);
}

int main() { return 0; }