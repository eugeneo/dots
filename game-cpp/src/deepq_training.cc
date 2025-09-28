#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
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

#include "src/deepq_loss.h"
#include "src/game.h"
#include "src/replay.h"
#include "uchen/training/kaiming_he.h"
#include "uchen/training/training.h"

using uchen::demo::Game;
using training_set = std::vector<
    std::pair<Game::QModel::input_t, uchen::learning::DeepQExpectation>>;

ABSL_FLAG(uint32_t, steps, 50, "Steps to play");
ABSL_FLAG(int, seed, 0, "Random seed");
ABSL_FLAG(bool, force, false, "Overwrite outputs");
ABSL_FLAG(std::string, input_params, "", "Input parameters");
ABSL_FLAG(std::string, output_params, "", "Output parameters");

uchen::demo::DotGameReplay SelfPlay(uint32_t steps, int seed) {
  uchen::demo::DotGameReplay replay;
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
    replay.RecordTurn(dots_game, step, random_index, player);
    player = 3 - player;
  }
  LOG(INFO) << "Done, " << steps
            << " steps, player 1 score: " << dots_game.player_score(1)
            << ", player 2 score " << dots_game.player_score(2);
  return replay;
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

std::optional<std::ifstream> OpenFileForRead(std::string_view filename) {
  namespace fs = std::filesystem;
  if (filename.empty()) {
    std::cerr << "Input file name is required.\n";
    return std::nullopt;
  }
  fs::path in_path(filename);
  if (!fs::exists(in_path)) {
    std::cerr << "File does not exist: " << filename << "\n";
    return std::nullopt;
  }
  std::ifstream ifs(std::string(filename).c_str(), std::ios::binary);
  if (!ifs) {
    std::cerr << "Cannot open file: " << filename << "\n";
    return std::nullopt;
  }
  return ifs;
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

std::optional<std::vector<uchen::demo::DotGameReplay>> ReadReplays(
    std::span<const char* const> files) {
  std::vector<uchen::demo::DotGameReplay> replays;
  for (auto* path : files) {
    auto is = OpenFileForRead(path);
    if (!is.has_value()) {
      LOG(FATAL) << "Can not open " << path;
      return std::nullopt;
    }
    auto replay = uchen::demo::DotGameReplay::Load(*is);
    if (!replay.has_value()) {
      return std::nullopt;
    }
    replays.emplace_back(std::move(replay).value());
  }
  return replays;
}

uchen::ModelParameters<Game::QModel> TrainingLoop(
    const uchen::ModelParameters<Game::QModel>& params,
    training_set training_batch) {
  uchen::training::Training training(&Game::model, params,
                                     uchen::learning::DeepQLoss{});
  uchen::training::TrainingData<Game::QModel::input_t,
                                uchen::learning::DeepQExpectation>
      data(std::make_move_iterator(training_batch.begin()),
           std::make_move_iterator(training_batch.end()));
  for (size_t generation = 1; training.Loss(data) > 0.0001; ++generation) {
    training = training.Generation(data, 0.001);
    if (generation > 500) {
      LOG(ERROR) << "Taking too long!";
    }
  }
  return training.parameters();
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
    if (l.size() != 3) {
      LOG(FATAL) << "File name required";
      return 1;
    }
    auto ofs = OpenFileForRead(l.back(), absl::GetFlag(FLAGS_force));
    if (!ofs.has_value()) {
      return 1;
    }
    auto replay =
        SelfPlay(absl::GetFlag(FLAGS_steps), absl::GetFlag(FLAGS_seed));
    if (!replay.Write(*ofs)) {
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
  } else if (verb == "train") {
    if (l.size() < 3) {
      LOG(FATAL) << "Replay files were not specified";
    }
    std::string starting = absl::GetFlag(FLAGS_input_params);
    std::string result = absl::GetFlag(FLAGS_output_params);
    LOG(INFO) << absl::Substitute(
        "Model parameters training, starting: $0, result: $1, using replays: "
        "$2",
        starting, result, absl::StrJoin(std::span(l).subspan(2), ", "));
    training_set training_batch;
    auto replays = ReadReplays(std::span(l).subspan(2));
    if (!replays.has_value()) {
      return 1;
    }
    size_t turns = 0;
    for (const auto& replay : std::move(replays).value()) {
      turns += replay.turns();
      std::vector batch = replay.ToTrainingSet(0.1);
      std::copy(std::make_move_iterator(batch.begin()),
                std::make_move_iterator(batch.end()),
                std::back_inserter(training_batch));
    }
    LOG(INFO) << absl::Substitute("$0 replays with $1 turns total. $2 samples",
                                  replays->size(), turns,
                                  training_batch.size());

    uchen::ModelParameters params = TrainingLoop(
        uchen::training::KaimingHeInitializedParameters(&Game::model),
        std::move(training_batch));
    return 1;
  }
  std::cerr << "Unknown verb: " << verb;
  return 1;
}