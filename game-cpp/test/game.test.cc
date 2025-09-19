#include "src/game.h"

#include <concepts>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gmock/gmock.h"

using uchen::demo::Game;

template <std::convertible_to<std::string_view>... Args>
Game BuildGame(const Args&... rows) {
  std::array<const std::string_view, sizeof...(Args)> input = {rows...};
  Game game(sizeof...(Args), input[0].size());
  for (size_t y = 0; y < sizeof...(Args); ++y) {
    CHECK_EQ(input[y].size(), input[0].size());
    for (size_t x = 0; x < input[y].size(); ++x) {
      char c = input[y][x];
      if (c != '.') {
        game.PlaceDot(y * input[y].size() + x, c - '0');
      }
    }
  }
  return game;
}

TEST(GameTest, SettingDots) {
  Game game(3, 3);
  EXPECT_THAT(game.field(), ::testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));
  game.PlaceDot(1, 1);
  game.PlaceDot(3, 1);
  game.PlaceDot(5, 1);
  game.PlaceDot(4, 2);
  EXPECT_THAT(game.polygons(), ::testing::IsEmpty());
  game.PlaceDot(7, 1);
  EXPECT_THAT(game.field(), ::testing::ElementsAre(0, 1, 0, 1, 2, 1, 0, 1, 0));
  EXPECT_THAT(game.polygons(), ::testing::UnorderedElementsAre(
                                   Game::Polygon({1, 3, 4, 5, 7}, 1, 1)));
}

TEST(GameTest, OneAway) {
  Game game = BuildGame("1.", ".1");
  EXPECT_THAT(game.PathBetween(0, 3), ::testing::ElementsAre(0, 3));
}

TEST(GameTest, ThreeAwayPath) {
  Game game = BuildGame("...1.", "..1.1", "..1.1", "....1");
  EXPECT_THAT(game.PathBetween(3, 9), ::testing::ElementsAre(3, 9));
  EXPECT_THAT(game.PathBetween(3, 14), ::testing::ElementsAre(3, 9, 14));
  EXPECT_THAT(game.PathBetween(3, 19), ::testing::ElementsAre(3, 9, 14, 19));
}

TEST(GameTest, Hook) {
  Game game = BuildGame("1.11", "1.1.", "11..");
  EXPECT_THAT(game.PathBetween(0, 3), ::testing::ElementsAre(0, 4, 9, 6, 3));
}

TEST(GameTest, FullyConnected) {
  Game game = BuildGame("111", "111", "111");
  EXPECT_THAT(game.PathBetween(0, 8), ::testing::ElementsAre(0, 4, 8));
}

TEST(GameTest, IgnoreTransition) {
  Game game = BuildGame(".1.", "1.1", ".1.");
  EXPECT_THAT(game.PathBetween(1, 3, {{1, 3}}),
              ::testing::ElementsAre(1, 5, 7, 3));
}

TEST(GameTest, DetectPolygon) {
  Game game = BuildGame("....", ".1..", "121.", ".1..");
  EXPECT_THAT(game.DetectPolygons(1, 1),
              ::testing::ElementsAre(Game::Polygon({5, 8, 9, 10, 13}, 1, 1)));
  EXPECT_THAT(game.DetectPolygons(0, 2),
              ::testing::ElementsAre(Game::Polygon({5, 8, 9, 10, 13}, 1, 1)));
  EXPECT_THAT(game.DetectPolygons(2, 2),
              ::testing::ElementsAre(Game::Polygon({5, 8, 9, 10, 13}, 1, 1)));
  EXPECT_THAT(game.DetectPolygons(1, 3),
              ::testing::ElementsAre(Game::Polygon({5, 8, 9, 10, 13}, 1, 1)));
}

TEST(GameTest, IgnoresEmptyPolygon) {
  Game game = BuildGame("....", ".1..", "1.1.", ".1..");
  EXPECT_THAT(game.DetectPolygons(1, 1), ::testing::IsEmpty());
}

TEST(GameTest, TwoPolygons) {
  Game game = BuildGame("1111", "1111", "121.", ".121", "1111");
  EXPECT_THAT(game.DetectPolygons(1, 3),
              ::testing::UnorderedElementsAre(
                  Game::Polygon({5, 8, 9, 10, 13}, 1, 1),
                  Game::Polygon({10, 13, 14, 15, 18}, 1, 1)));
}

TEST(GameTest, TwoPolygonsOneNotFilled) {
  Game game = BuildGame("1111", "1111", "121.", ".1.1", "1111");
  EXPECT_THAT(
      game.DetectPolygons(1, 3),
      ::testing::UnorderedElementsAre(Game::Polygon({5, 8, 9, 10, 13}, 1, 1)));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  return RUN_ALL_TESTS();
}