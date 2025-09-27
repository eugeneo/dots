#include <cstddef>
#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "src/game.h"

ABSL_FLAG(uint32_t, steps, 50, "Steps to play");
ABSL_FLAG(std::string, output, "", "Output file name");
ABSL_FLAG(int, seed, 0, "Random seed");

struct SelfPlayTurnRecord {
  std::vector<size_t> dots_our;
  std::vector<size_t> dots_opponent;
  std::vector<size_t> captured_our;
  std::vector<size_t> captured_opponent;
  int move;
  size_t score_our;
  size_t score_opponent;
};

SelfPlayTurnRecord RecordTurn(const uchen::demo::Game& game, int move,
                              int player) {
  std::vector<size_t> dots_our, dots_opponent, captured_our, captured_opponent;
  return SelfPlayTurnRecord{.move = move,
                            .score_our = game.player_score(player),
                            .score_opponent = game.player_score(3 - player)};
}

int SelfPlay(uint32_t steps, int seed) {
  std::random_device rd;
  std::mt19937 gen(rd());
  if (seed != 0) {
    gen.seed(seed);
  }

  uchen::demo::Game dots_game(64, 64);
  // Always the first turn
  dots_game.PlaceDot(31 * 64 + 31, 1);
  LOG(INFO) << "Self-playing for " << steps << " steps";

  int player = 2;

  for (size_t step = 0; step < 100; ++step) {
    std::vector<int> good_indexes = dots_game.GetGoodAutoplayerIndexes();
    std::uniform_int_distribution<> dis(0, good_indexes.size() - 1);
    size_t ind = dis(gen);
    int random_index = good_indexes[ind];
    dots_game.PlaceDot(random_index, player);
    LOG(INFO) << "Step " << step << " Random index from good_indexes: " << ind
              << " player " << player
              << " score: " << dots_game.player_score(player)
              << " open indexes: " << good_indexes.size();
    std::string field_str = absl::StrJoin(
        dots_game.field(), "", [](std::string* result, uint8_t v) {
          absl::StrAppend(result, v == 0 ? " " : absl::StrCat(v));
        });
    bool met_non_empty = false;
    int empties = 0;
    for (std::string_view sv : absl::StrSplit(field_str, absl::ByLength(64))) {
      std::string s{sv};
      absl::StripAsciiWhitespace(&s);
      if (!s.empty()) {
        met_non_empty = true;
        for (int i = 0; i < empties; ++i) {
          LOG(INFO) << " ";
        }
        LOG(INFO) << sv;
        empties = 0;
      } else if (met_non_empty) {
        empties += 1;
      }
    }

    player = 3 - player;
  }
  LOG(INFO) << "Done";
  return 0;
}

int main(int argc, char** argv) {
  auto l = absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  if (l.size() == 1) {
    std::cerr << "Verb is missing";
    return 1;
  }
  if (std::string_view(l[1]) == "selfplay") {
    return SelfPlay(absl::GetFlag(FLAGS_steps), absl::GetFlag(FLAGS_seed));
  }
  std::cerr << "Unknown verb: " << l[1];
  return 1;
  // ConvolutionInput<4, 64, 64> input;
  // uchen::ModelParameters params(&ConvQNetwork);
  // auto r = ConvQNetwork(input, params);
  // uchen::training::Training training(&ConvQNetwork, params);
  // uchen::training::TrainingData<QModel::input_t, QModel::output_t> data = {};
  // for (size_t generation = 1; training.Loss(data) > 0.0001; ++generation) {
  //   training = training.Generation(data, 0.001);
  //   if (generation > 500) {
  //     LOG(ERROR) << "Taking too long!";
  //   }
  // }
}