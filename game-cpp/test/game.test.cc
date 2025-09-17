#include "src/game.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using uchen::demo::Game;

TEST(GameTest, SettingDots) {
  Game game(3, 3);
  EXPECT_THAT(game.field(), ::testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));
  game.GameTurn(1, 1);
  game.GameTurn(3, 1);
  game.GameTurn(5, 1);
  EXPECT_THAT(game.field(), ::testing::ElementsAre(0, 1, 0, 1, 0, 1, 0, 0, 0));
  EXPECT_EQ(game.player_score(1), 0);
  game.GameTurn(7, 1);
  EXPECT_THAT(game.field(),
              ::testing::ElementsAre(0, 0x41, 0, 0x41, 0x41, 0x41, 0, 0x41, 0));
  EXPECT_EQ(game.player_score(1), 5);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}