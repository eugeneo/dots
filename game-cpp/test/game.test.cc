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

Game::PlayerOverlay ParseOverlay(absl::string_view field) {
  std::vector<std::string_view> lines = absl::StrSplit(
      absl::StripPrefix(absl::StripSuffix(field, "|"), "|"), "|");
  CHECK(!lines.empty());
  Game::PlayerOverlay overlay{lines[0].length(), lines.size()};
  for (size_t y = 0; y < lines.size(); ++y) {
    for (size_t x = 0; x < lines[y].length(); ++x) {
      overlay.set_dot(y * overlay.width() + x, lines[y][x] == 'x');
    }
  }
  return overlay;
}

TEST(GameTest, SettingDots) {
  Game game(3, 3);
  EXPECT_THAT(game.field(), ::testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));
  game.PlaceDot(1, 1);
  game.PlaceDot(3, 1);
  game.PlaceDot(5, 1);
  game.PlaceDot(4, 2);
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|...|...|...|"));
  EXPECT_EQ(game.player_overlay(1).captured(), 0);
  game.PlaceDot(7, 1);
  EXPECT_THAT(game.field(), ::testing::ElementsAre(0, 1, 0, 1, 2, 1, 0, 1, 0));
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|.x.|xxx|.x.|"));
  EXPECT_EQ(game.player_overlay(1).captured(), 1);
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
  game.FillPolygons(1, 1);
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|....|.x.|xxx|.x|"));
  game = BuildGame("....", ".1..", "121.", ".1..");
  game.FillPolygons(0, 2);
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|....|.x.|xxx|.x|"));
  game = BuildGame("....", ".1..", "121.", ".1..");
  game.FillPolygons(2, 2);
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|....|.x.|xxx|.x|"));
  game = BuildGame("....", ".1..", "121.", ".1..");
  game.FillPolygons(1, 3);
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|....|.x.|xxx|.x|"));
}

TEST(GameTest, IgnoresEmptyPolygon) {
  Game game = BuildGame("....", ".1..", "1.1.", ".1..");
  game.FillPolygons(1, 3);
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|....|.|.|.|"));
}

TEST(GameTest, TwoPolygons) {
  Game game = BuildGame("1111", "1111", "121.", ".121", "1111");
  game.FillPolygons(1, 3);
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|....|.x..|xxx.|.xxx|.xx.|"));
}

TEST(GameTest, TwoPolygonsOneNotFilled) {
  Game game = BuildGame("1111", "1.11", "121.", ".1.1", "1111");
  game.FillPolygons(1, 3);
  EXPECT_EQ(game.player_overlay(1), ParseOverlay("|.x..|xxx.|xxx.|.x|.|"));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  return RUN_ALL_TESTS();
}