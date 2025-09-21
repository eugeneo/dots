#include <cstddef>
#include <cstdint>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "absl/log/log.h" // IWYU pragma: keep for logging macros
#include "src/game.h"

namespace {
using uchen::demo::Game;

struct Region {
  std::string shape;
  int player;
};

std::string ToSvgPath(const Game::Polygon &polygon, int grid_size) {
  std::vector<std::string> parts = {
      absl::StrCat("M ", polygon.x * grid_size + grid_size / 2, ",",
                   polygon.y * grid_size + grid_size / 2)};
  for (auto d : polygon.outline) {
    auto [dx, dy] = Game::Polygon::kDirections[static_cast<size_t>(d)];
    parts.push_back(absl::StrCat("l ", dx * grid_size, ",", dy * grid_size));
  }
  parts.push_back("Z");
  return absl::StrJoin(parts, " ");
}

class GameWrapper {
public:
  GameWrapper(int h, int w) : game_(h, w) {
    field_ = emscripten::val(emscripten::typed_memory_view(
        game_.field().size(), game_.field().data()));
  }

  emscripten::val GetField() const { return field_; }

  void GameTurn(size_t index, uint8_t player_id) {
    if (game_.PlaceDot(index, player_id)) {
      UpdateRegions();
    }
  }

  size_t player_score(uint8_t player_id) const {
    return game_.player_score(player_id + 1);
  }

  std::vector<Region> get_regions() const { return regions_; }

private:
  void UpdateRegions() {
    regions_.clear();
    for (const auto &polygon : game_.polygons()) {
      regions_.push_back(Region{.shape = ToSvgPath(polygon, grid_size_),
                                .player = polygon.player - 1});
    }
  }

  Game game_;
  emscripten::val field_;
  std::vector<Region> regions_;
  int grid_size_ = 24;
};

} // namespace

EMSCRIPTEN_BINDINGS(dots_logic) {
  emscripten::register_vector<Region>("RegionVector");
  emscripten::class_<GameWrapper>("Game")
      .constructor<int, int>()
      .function("field", &GameWrapper::GetField)
      .function("doTurn", &GameWrapper::GameTurn)
      .function("playerScore", &GameWrapper::player_score)
      .function("regions", &GameWrapper::get_regions);
  emscripten::class_<Region>("Region")
      .property("shape", &Region::shape)
      .property("player", &Region::player);
}

int main() { return 0; }