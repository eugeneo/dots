#include "src/game.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace uchen::demo {
namespace {

using Polygon = Game::Polygon;
using Direction = Polygon::Direction;

constexpr std::array kNextDirection = {
    Direction::kW, Direction::kNw, Direction::kN, Direction::kNe,
    Direction::kE, Direction::kSe, Direction::kS, Direction::kSw,
    Direction::kW, Direction::kNw, Direction::kN, Direction::kNe};

// Generators in C++23 will make this much better.
absl::InlinedVector<size_t, 8> SurroundingIndexes(size_t index, size_t width,
                                                  size_t height) {
  int x = index % width;
  int y = index / width;
  absl::InlinedVector<size_t, 8> indexes;
  for (auto [dx, dy] : Polygon::kDirections) {
    int nx = x + dx;
    int ny = y + dy;
    if (nx < 0 || nx >= width) {
      continue;
    }
    if (ny < 0 || ny >= height) {
      continue;
    }
    indexes.push_back(nx + ny * width);
  }
  return indexes;
}

std::vector<Direction> OutlineRegion(size_t start_x, size_t start_y,
                                     const Game::PlayerOverlay& overlay,
                                     size_t height, size_t width,
                                     std::vector<bool>& visited) {
  size_t x = start_x;
  size_t y = start_y;
  std::vector<Direction> outline;
  Direction dir = Direction::kE;
  size_t max_steps = 5000;
  do {
    if (max_steps-- == 0) {
      LOG(FATAL) << "OutlineRegion: too many steps";
      break;
    }
    std::span nextDirs = std::span<const Direction>(kNextDirection)
                             .subspan(static_cast<size_t>(dir), 5);
    for (auto d : nextDirs) {
      auto [dx, dy] = Polygon::kDirections[static_cast<size_t>(d)];
      size_t nx = x + dx;
      size_t ny = y + dy;
      if (nx < 0 || nx >= width) {
        continue;
      }
      if (ny < 0 || ny >= height) {
        continue;
      }
      size_t nind = nx + ny * width;
      if (overlay.get_dot(nind)) {
        dir = d;
        outline.push_back(d);
        visited[nind] = true;
        x = nx;
        y = ny;
        break;
      }
    }
  } while (x != start_x || y != start_y);
  return outline;
}

std::vector<Game::Polygon> UpdateRegions(const Game& game) {
  std::vector<Game::Polygon> polygons;
  int player_id = 0;
  size_t size = game.field().size();
  std::vector<bool> visited(size, false);
  size_t width = game.width();
  size_t height = size / width;
  std::unordered_set<int> outlined_regions;
  for (const auto& overlay : game.player_overlays()) {
    std::fill(visited.begin(), visited.end(), false);
    player_id += 1;
    for (size_t i = 0; i < size; ++i) {
      if (visited[i]) {
        continue;
      }
      int region = overlay.get_dot(i);
      if (region == 0 || outlined_regions.contains(region)) {
        continue;
      }
      outlined_regions.insert(region);
      polygons.push_back(Game::Polygon{
          .x = i % width,
          .y = i / width,
          .outline = OutlineRegion(i % width, i / width, overlay, height, width,
                                   visited),
          .player = player_id,
      });
    }
  }
  return polygons;
}

}  // namespace

Game::Game(int height, int width)
    : width_(width),
      field_(height * width, 0),
      valid_moves(height * width, Game::CellForMove::kFar) {
  CHECK_GT(height, 0);
  CHECK_GT(width, 0);
}

bool Game::PlaceDot(size_t index, uint8_t player_id) {
  if (player_at(index) != 0) {
    return false;
  }
  int x = index % width_;
  int y = index / width_;
  set_dot(x, y, player_id);
  bool filled_polygon = FillPolygons(x, y);
  if (filled_polygon) {
    return false;
    polygons_ = UpdateRegions(*this);
  }
  valid_moves[index] = CellForMove::kOccupied;
  for (int kx = std::max(x - Game::kGoodMoveRange, 0);
       kx < std::min(x + Game::kGoodMoveRange, width_); ++kx) {
    for (int ky = std::max(y - Game::kGoodMoveRange, 0);
         ky < std::min(y + kGoodMoveRange, static_cast<int>(index / width_));
         ++ky) {
      size_t index = kx + ky * width_;
      if (valid_moves[index] == CellForMove::kFar) {
        valid_moves[index] = CellForMove::kGood;
      }
    }
  }
  return filled_polygon;
}

absl::InlinedVector<size_t, 64 * 64> Game::PathBetween(
    size_t start, size_t end,
    const std::set<std::pair<size_t, size_t>>& ignored_transitions) const {
  int player_id = player_at(start);
  if (start >= field_.size() || end >= field_.size() || player_id == 0 ||
      player_at(end) != player_id) {
    return {};
  }
  absl::InlinedVector<std::optional<size_t>, kBufferSize> previous(
      field_.size(), std::nullopt);
  previous[start] = std::numeric_limits<size_t>::max();
  const size_t height = field_.size() / width_;
  std::deque<size_t> to_visit = {start};
  while (!to_visit.empty()) {
    size_t index = to_visit.front();
    to_visit.pop_front();
    for (size_t ni : SurroundingIndexes(index, width_, height)) {
      if (player_at(ni) != player_id) {
        continue;
      }
      if (Captured(ni)) {
        continue;
      }
      if (ignored_transitions.contains({index, ni}) ||
          ignored_transitions.contains({ni, index})) {
        continue;
      }
      if (ni == end) {
        absl::InlinedVector<size_t, 64 * 64> result = {ni};
        for (size_t i = index; i != std::numeric_limits<size_t>::max();
             i = previous[i].value()) {
          result.push_back(i);
        }
        std::reverse(result.begin(), result.end());
        return result;
      }
      if (previous[ni].has_value()) {
        continue;
      }
      to_visit.push_back(ni);
      previous[ni] = index;
    }
  }
  return {};
}

bool Game::FillPolygons(int x, int y) {
  size_t index = x + y * width_;
  CHECK_LT(index, field_.size());
  int player = player_at(index);
  CHECK_NE(player, 0);
  // We "break" connections so we don't report same polygons repeatedly.
  std::set<std::pair<size_t, size_t>> ignored_transitions;
  bool updated = false;
  for (size_t ni :
       SurroundingIndexes(x + y * width_, width_, field_.size() / width_)) {
    if (player_at(ni) != player) {
      continue;
    }
    ignored_transitions.emplace(index, ni);
    auto path = PathBetween(index, ni, ignored_transitions);
    if (path.empty()) {
      continue;
    }
    FillPath(path, player);
    updated = true;
  }
  return updated;
}

void Game::FillPath(std::span<const size_t> path, int player_id) {
  size_t top = std::numeric_limits<size_t>::max(), bottom = 0,
         left = std::numeric_limits<size_t>::max(), right = 0;
  for (size_t i : path) {
    top = std::min(top, i / width_);
    bottom = std::max(bottom, i / width_);
    left = std::min(left, i % width_);
    right = std::max(right, i % width_);
  }
  size_t w = right - left + 1;
  size_t h = bottom - top + 1;
  Game::Grid grid(left, top, w, h);
  // 1. Fill edges with outside.
  for (size_t i : path) {
    grid.mark_inside((i / width_ - top) * w + (i % width_ - left));
  }
  // 2. Mark outside.
  std::deque<size_t> to_visit;
  for (size_t x = 0; x < w; ++x) {
    to_visit.push_back(x);
    to_visit.push_back(x + (bottom - top) * w);
  }
  for (size_t y = 0; y < bottom - top + 1; ++y) {
    to_visit.push_back(y * w);
    to_visit.push_back(y * w + (right - left));
  }
  while (!to_visit.empty()) {
    size_t index = to_visit.front();
    to_visit.pop_front();
    if (!grid.is_unknown(index)) {
      continue;
    }
    grid.mark_outside(index);
    size_t x = index % w;
    size_t y = index / w;
    if (x > 0) {
      to_visit.push_back(index - 1);
    }
    if (x < right - left) {
      to_visit.push_back(index + 1);
    }
    if (y > 0) {
      to_visit.push_back(index - w);
    }
    if (y < bottom - top) {
      to_visit.push_back(index + w);
    }
  }
  player_overlay(player_id).MarkRegion(grid, *this);
}

bool Game::Captured(size_t index) const {
  int id = player_at(index);
  if (id == 0) {
    return false;
  }
  for (size_t i = 0; i < overlays_.size(); ++i) {
    if (i == id - 1) {
      continue;
    }
    if (overlays_[i].captured(index)) {
      return true;
    }
  }
  return false;
}

void Game::PlayerOverlay::MarkRegion(const Grid& grid, const Game& game) {
  // 3. Fill inside, count captured dots.
  std::vector<bool> data;
  size_t captured = 0;
  int left = grid.x();
  int top = grid.y();
  int w = grid.width();
  int h = grid.height();
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      data.emplace_back(!grid.is_outside(x + y * w));
      if (!grid.is_unknown(x + y * w)) {
        continue;
      }
      size_t ind = x + left + (y + top) * width_;
      int p = game.player_at(ind);
      if (p != 0 && p != player_id_) {
        set_captured(ind);
        ++captured;
      }
    }
  }
  if (captured == 0) {
    return;
  }
  std::unordered_set<int> regions_to_merge;
  int new_region_id = next_region_id_++;
  for (size_t i = 0; i < data.size(); ++i) {
    size_t index = i % w + left + (i / w + top) * width();
    int region = get_dot(index);
    if (region != 0) {
      regions_to_merge.insert(region);
    }
    if (data[i]) {
      set_dot(index, new_region_id);
    }
  }
  if (regions_to_merge.empty()) {
    return;
  }
  for (size_t i = 0; i < data_.size(); ++i) {
    int region = get_dot(i);
    if (region != 0 && regions_to_merge.contains(region)) {
      set_dot(i, new_region_id);
    }
  }
}

std::vector<int> Game::GetGoodAutoplayerIndexes() const {
  std::vector<int> indexes;
  for (int i = 0; i < valid_moves.size(); ++i) {
    if (valid_moves[i] == CellForMove::kGood) {
      indexes.push_back(i);
    }
  }
  return indexes;
}

size_t Game::SuggestMove() const {
  QModel::input_t input;
  uchen::ModelParameters par{&model};
  auto output = model(input, par);
  size_t r;
  float max = std::numeric_limits<float>::min();
  for (size_t i = 0; i < output.size(); ++i) {
    if (output[i] > max) {
      r = i;
      max = output[i];
    }
  }
  return r;
}

}  // namespace uchen::demo