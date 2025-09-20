#include "src/game.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"  // IWYU pragma: keep for LOG(INFO)

namespace uchen::demo {
namespace {

constexpr std::array<std::pair<int, int>, 8> kDirections = {{
    {-1, -1},
    {0, -1},
    {1, -1},
    {-1, 0},
    {1, 0},
    {-1, 1},
    {0, 1},
    {1, 1},
}};

// Generators in C++23 will make this much better.
absl::InlinedVector<size_t, 8> SurroundingIndexes(size_t index, size_t width,
                                                  size_t height) {
  int x = index % width;
  int y = index / width;
  absl::InlinedVector<size_t, 8> indexes;
  for (auto [dx, dy] : kDirections) {
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

std::vector<bool> ParseElements(std::string_view data) {
  std::vector<bool> result;
  for (char c : data) {
    if (c != '|') {
      result.emplace_back(c == 'x');
    }
  }
  return result;
}

}  // namespace

Game::Game(int height, int width) : width_(width), field_(height * width, 0) {
  CHECK_GT(height, 0);
  CHECK_GT(width, 0);
}

void Game::PlaceDot(size_t index, uint8_t player_id) {
  int x = index % width_;
  int y = index / width_;
  set_dot(x, y, player_id);
  auto polygons = DetectPolygons(x, y);
  std::copy(polygons.begin(), polygons.end(), std::back_inserter(polygons_));
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

std::unordered_set<Game::Polygon> Game::DetectPolygons(int x, int y) const {
  size_t index = x + y * width_;
  CHECK_LT(index, field_.size());
  int player = player_at(index);
  CHECK_NE(player, 0);
  std::unordered_set<Game::Polygon> polygons;
  // We "break" connections so we don't report same polygons repeatedly.
  std::set<std::pair<size_t, size_t>> ignored_transitions;
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
    auto polygon = PolygonFromPath(path, player);
    if (polygon.has_value()) {
      polygons.emplace(std::move(polygon).value());
    }
  }
  return polygons;
}

std::optional<Game::Polygon> Game::PolygonFromPath(std::span<const size_t> path,
                                                   int player_id) const {
  enum class State { kUnknown, kInside, kOutside };
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
  absl::InlinedVector<State, Game::kBufferSize> grid(h * w, State::kUnknown);
  // 1. Fill edges with outside.
  for (size_t i : path) {
    grid[(i / width_ - top) * w + (i % width_ - left)] = State::kInside;
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
    if (grid[index] != State::kUnknown) {
      continue;
    }
    grid[index] = State::kOutside;
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
  // 3. Fill inside, count captured dots.
  size_t captured = 0;
  std::vector<bool> data;
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      State s = grid[x + y * w];
      data.emplace_back(s != State::kOutside);
      if (s != State::kUnknown) {
        continue;
      }
      int p = player_at(x + left + (y + top) * width_);
      if (p != 0 && p != player_id) {
        captured += 1;
      }
    }
  }
  if (captured == 0) {
    return std::nullopt;
  }
  absl::InlinedVector<size_t, Game::kBufferSize> polygon;
  for (size_t i = 0; i < grid.size(); ++i) {
    if (grid[i] == State::kInside) {
      size_t x = i % w;
      size_t y = i / w;
      polygon.push_back(x + left + (y + top) * width_);
    }
  }
  return Game::Polygon(left, top, w, data, captured, player_id);
}

Game::Polygon::Polygon(int x, int y, int width, std::string_view elements,
                       size_t captures, int player)
    : Polygon(x, y, width, ParseElements(elements), captures, player) {}

std::string Game::Polygon::StrData() const {
  std::ostringstream result;
  result << "|";
  for (size_t start = 0; start < data_.size(); start += width_) {
    result << absl::StrJoin(
                  data_.begin() + start, data_.begin() + start + width_, "",
                  [](auto* s, bool b) { absl::StrAppend(s, b ? "x" : "."); })
           << "|";
  }
  return result.str();
}

}  // namespace uchen::demo