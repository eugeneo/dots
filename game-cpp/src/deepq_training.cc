#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "src/game.h"
#include "uchen/training/kaiming_he.h"

using uchen::demo::Game;

ABSL_FLAG(uint32_t, steps, 50, "Steps to play");
ABSL_FLAG(std::string, output, "", "Output file name");
ABSL_FLAG(int, seed, 0, "Random seed");
ABSL_FLAG(bool, force, false, "Overwrite outputs");
ABSL_FLAG(std::string, input_params, "", "Input parameters");
ABSL_FLAG(std::string, output_params, "", "Output parameters");

struct SelfPlayTurnRecord {
  std::vector<uint32_t> dots_our;
  std::vector<uint32_t> dots_opponent;
  std::vector<uint32_t> captured_our;
  std::vector<uint32_t> captured_opponent;
  uint32_t move;
  uint32_t score_our;
  uint32_t score_opponent;
  int step;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SelfPlayTurnRecord& record) {
    sink.Append(absl::Substitute(
        "S:$7 M:$0 P1:$1 P2:$2 {$3}{$4}{$5}{$6}", record.move, record.score_our,
        record.score_opponent, absl::StrJoin(record.dots_our, ","),
        absl::StrJoin(record.dots_opponent, ","),
        absl::StrJoin(record.captured_our, ","),
        absl::StrJoin(record.captured_opponent, ","), record.step));
  }
};

std::vector<uint32_t> RecordCaptures(
    std::span<const Game::PlayerOverlay> overlays, int player) {
  if (overlays.size() <= player) {
    return {};
  }
  std::vector<uint32_t> capt;
  const auto& overlay = overlays[player];
  for (uint32_t i = 0; i < overlay.height() * overlay.width(); ++i) {
    if (overlay.captured(i)) {
      capt.emplace_back(i);
    }
  }
  return capt;
}

SelfPlayTurnRecord RecordTurn(const Game& game, int step, uint32_t move,
                              uint32_t player) {
  SelfPlayTurnRecord result = {
      .move = move,
      .score_our = game.player_score(player),
      .score_opponent = game.player_score(3 - player),
      .captured_our = RecordCaptures(game.player_overlays(), player - 1),
      .captured_opponent = RecordCaptures(game.player_overlays(), 2 - player),
      .step = step};
  std::span field = game.field();
  for (uint32_t i = 0; i < field.size(); ++i) {
    if (field[i] == player) {
      result.dots_our.emplace_back(i);
    } else if (field[i] != 0) {
      result.dots_opponent.emplace_back(i);
    }
  }

  return result;
}

std::array<std::vector<SelfPlayTurnRecord>, 2> SelfPlay(uint32_t steps,
                                                        int seed) {
  std::array<std::vector<SelfPlayTurnRecord>, 2> logs;
  std::vector<SelfPlayTurnRecord> p2;
  std::random_device rd;
  std::mt19937 gen(rd());
  if (seed != 0) {
    gen.seed(seed);
  }

  Game dots_game(64, 64);
  // Always the first turn
  dots_game.PlaceDot(31 * 64 + 31, 1);
  LOG(INFO) << "Self-playing for " << steps << " steps";

  int player = 2;

  for (size_t step = 0; step < steps; ++step) {
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
    logs[player - 1].emplace_back(
        RecordTurn(dots_game, step, random_index, player));
    LOG(INFO) << logs[player - 1].back();
    player = 3 - player;
  }
  LOG(INFO) << "Done, " << steps
            << " steps, player 1 score: " << dots_game.player_score(1)
            << ", player 2 score " << dots_game.player_score(2);
  return logs;
}

bool Write(std::ofstream& out, std::span<const SelfPlayTurnRecord> records) {
  for (const auto& record : records) {
    // Write step
    out.write(reinterpret_cast<const char*>(&record.step), sizeof(record.step));
    // Write move
    out.write(reinterpret_cast<const char*>(&record.move), sizeof(record.move));
    // Write score_our
    out.write(reinterpret_cast<const char*>(&record.score_our),
              sizeof(record.score_our));
    // Write score_opponent
    out.write(reinterpret_cast<const char*>(&record.score_opponent),
              sizeof(record.score_opponent));

    // Write dots_our
    uint32_t dots_our_size = record.dots_our.size();
    out.write(reinterpret_cast<const char*>(&dots_our_size),
              sizeof(dots_our_size));
    if (dots_our_size > 0) {
      out.write(reinterpret_cast<const char*>(record.dots_our.data()),
                dots_our_size * sizeof(uint32_t));
    }

    // Write dots_opponent
    uint32_t dots_opponent_size = record.dots_opponent.size();
    out.write(reinterpret_cast<const char*>(&dots_opponent_size),
              sizeof(dots_opponent_size));
    if (dots_opponent_size > 0) {
      out.write(reinterpret_cast<const char*>(record.dots_opponent.data()),
                dots_opponent_size * sizeof(uint32_t));
    }

    // Write captured_our
    uint32_t captured_our_size = record.captured_our.size();
    out.write(reinterpret_cast<const char*>(&captured_our_size),
              sizeof(captured_our_size));
    if (captured_our_size > 0) {
      out.write(reinterpret_cast<const char*>(record.captured_our.data()),
                captured_our_size * sizeof(uint32_t));
    }

    // Write captured_opponent
    uint32_t captured_opponent_size = record.captured_opponent.size();
    out.write(reinterpret_cast<const char*>(&captured_opponent_size),
              sizeof(captured_opponent_size));
    if (captured_opponent_size > 0) {
      out.write(reinterpret_cast<const char*>(record.captured_opponent.data()),
                captured_opponent_size * sizeof(uint32_t));
    }

    if (!out) {
      return false;
    }
  }
  return true;
}

std::optional<std::ofstream> OpenFileForRead(std::string_view filename,
                                             bool force) {
  namespace fs = std::filesystem;
  if (filename.empty()) {
    std::cerr << "Output file name is required.\n";
    return std::nullopt;
  }
  fs::path out_path(filename);
  if (fs::exists(out_path) && !absl::GetFlag(FLAGS_force)) {
    std::cerr << "File already exists: " << filename << "\n";
    return std::nullopt;
  }
  std::error_code ec;
  std::ofstream ofs(std::string(filename).c_str());
  if (!ofs || ec) {
    std::cerr << "Cannot create file: " << filename << "\n";
    return std::nullopt;
  }
  return ofs;
}

template <size_t Parameters>
bool WriteLayerParameters(const uchen::Parameters<Parameters>& p,
                          std::ofstream& stream, size_t* out_wrote) {
  if constexpr (Parameters == 0) {
    return true;
  } else {
    std::span<const float> span = p;
    stream.write(reinterpret_cast<const char*>(span.data()),
                 span.size() * sizeof(float));
    if (!stream) {
      return false;
    }
    *out_wrote += span.size() * sizeof(float);
    return true;
  }
}

template <typename Model, size_t... S>
std::optional<size_t> WriteParameters(
    const uchen::ModelParameters<Model>& parameters, std::ofstream& stream,
    std::index_sequence<S...> /* seq */) {
  size_t wrote = 0;
  if (true &&
      (... && WriteLayerParameters(parameters.template layer_parameters<S>(),
                                   stream, &wrote))) {
    return wrote;
  } else {
    return std::nullopt;
  }
}

int main(int argc, char** argv) {
  auto l = absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  if (l.size() == 1) {
    std::cerr << "Verb is missing";
    return 1;
  }
  std::string_view verb = l[1];
  if (verb == "selfplay") {
    auto ofs = OpenFileForRead(absl::GetFlag(FLAGS_output),
                               absl::GetFlag(FLAGS_force));
    if (!ofs.has_value()) {
      return 1;
    }
    auto [p1, p2] =
        SelfPlay(absl::GetFlag(FLAGS_steps), absl::GetFlag(FLAGS_seed));
    if (!Write(*ofs, p1) || !Write(*ofs, p2)) {
      return 1;
    }
    return 0;
  } else if (verb == "init_parameters") {
    auto ofs = OpenFileForRead(absl::GetFlag(FLAGS_output_params),
                               absl::GetFlag(FLAGS_force));
    if (!ofs.has_value()) {
      return 1;
    }
    auto parameters =
        uchen::training::KaimingHeInitializedParameters(&Game::model);
    auto wrote = WriteParameters(
        parameters, *ofs, std::make_index_sequence<Game::QModel::kLayers>());
    if (!wrote.has_value()) {
      return 1;
    }
    LOG(INFO) << "Wrote " << *wrote << " bytes";
    return 0;
  }
  std::cerr << "Unknown verb: " << verb;
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