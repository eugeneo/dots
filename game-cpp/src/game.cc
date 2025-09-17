#include "src/game.h"
#include <cstddef>

namespace uchen::demo {

namespace {
const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

// Helper to check if a point is inside a polygon using ray casting
bool point_in_polygon(int x, int y,
                      const std::vector<std::pair<int, int>> &poly) {
  int cnt = 0;
  size_t n = poly.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    int xi = poly[i].first, yi = poly[i].second;
    int xj = poly[j].first, yj = poly[j].second;
    if (((yi > y) != (yj > y)) &&
        (x < (xj - xi) * (y - yi) / ((yj - yi) ? (yj - yi) : 1) + xi))
      cnt++;
  }
  return cnt % 2 == 1;
}

// DFS to find a cycle (polygon) including (start_x, start_y)
bool dfs_cycle(const uchen::demo::Game &game, int x, int y, int from_x,
               int from_y, int start_x, int start_y, int player_id,
               std::vector<std::vector<bool>> &visited,
               std::vector<std::pair<int, int>> &path,
               std::vector<std::pair<int, int>> &polygon, int depth) {
  visited[y][x] = true;
  path.push_back({x, y});
  size_t width = game.width();
  size_t height = game.field().size() / width;
  for (int dir = 0; dir < 8; ++dir) {
    int nx = x + dx[dir], ny = y + dy[dir];
    if (nx < 0 || ny < 0 || nx >= (int)width || ny >= (int)height)
      continue;
    if (nx == from_x && ny == from_y)
      continue; // Don't go back to previous
    if (game.player_at(nx, ny) != player_id)
      continue;
    if (nx == start_x && ny == start_y && depth >= 4) {
      polygon = path;
      path.pop_back();
      visited[y][x] = false;
      return true;
    }
    if (!visited[ny][nx]) {
      if (dfs_cycle(game, nx, ny, x, y, start_x, start_y, player_id, visited,
                    path, polygon, depth + 1)) {
        path.pop_back();
        visited[y][x] = false;
        return true;
      }
    }
  }
  path.pop_back();
  visited[y][x] = false;
  return false;
}

} // namespace

Game::Game(size_t height, size_t width)
    : width_(width), field_(height * width, 0) {}

void Game::GameTurn(size_t index, uint8_t player_id) {
  int x = index % width_;
  int y = index / width_;
  set_dot(x, y, player_id, false);
  // Detect the polygon and fill it. Update player score - every enemy dot gives
  // 10 points, empty dot gives 1 point.

  // Detect polygon including (x, y)
  std::vector<std::pair<int, int>> polygon;
  size_t height = field_.size() / width_;
  std::vector<std::vector<bool>> visited(height,
                                         std::vector<bool>(width_, false));
  std::vector<std::pair<int, int>> path;
  bool found = dfs_cycle(*this, x, y, -1, -1, x, y, player_id, visited, path,
                         polygon, 1);
  if (found) {
    // Mark all dots in the polygon with polygon bit
    for (const auto &[px, py] : polygon) {
      set_dot(px, py, player_id, true);
    }
    // Fill inside the polygon and update score
    size_t filled = 0;
    for (size_t fy = 0; fy < height; ++fy) {
      for (size_t fx = 0; fx < width_; ++fx) {
        if (point_in_polygon(fx, fy, polygon)) {
          set_dot(fx, fy, player_id, true);
          ++filled;
        }
      }
    }
    player_scores_[player_id] += filled;
  }
}

} // namespace uchen::demo