#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <ostream>
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
using uchen::ModelParameters;
using uchen::training::ParameterGradients;

ABSL_FLAG(uint32_t, steps, 50, "Steps to play");
ABSL_FLAG(int, seed, 0, "Random seed");
ABSL_FLAG(bool, force, false, "Overwrite outputs");
ABSL_FLAG(std::string, input_params, "", "Input parameters");
ABSL_FLAG(std::string, output_params, "", "Output parameters");
ABSL_FLAG(float, model_play, 1.f, "How often the model plays");

template <typename M>
class AdamOptimizer {
 public:
  AdamOptimizer(float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
                ParameterGradients<M> m = {}, ParameterGradients<M> v = {},
                size_t step = 1)
      : beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        m_(std::move(m)),
        v_(std::move(v)),
        step_(step) {}

  std::pair<ModelParameters<M>, AdamOptimizer<M>> operator()(
      const ModelParameters<M>& params, const ParameterGradients<M>& grads,
      size_t batch_size, float learning_rate) const {
    CHECK_GE(step_, 1);
    ParameterGradients<M> m;
    ParameterGradients<M> v;
    ParameterGradients<M> deltas;  // update buffer

    const float lr_scaled = learning_rate;
    const float bias_correction1 =
        1.0f - std::pow(beta1_, static_cast<float>(step_));
    const float bias_correction2 =
        1.0f - std::pow(beta2_, static_cast<float>(step_));

    for (size_t i = 0; i < grads.size(); ++i) {
      float g = grads[i];

      // update biased moments
      m[i] = beta1_ * m_[i] + (1.0f - beta1_) * g;
      v[i] = beta2_ * v_[i] + (1.0f - beta2_) * (g * g);

      // bias correction
      float m_hat = m[i] / bias_correction1;
      float v_hat = v[i] / bias_correction2;

      // parameter delta
      deltas[i] = lr_scaled * m_hat / (std::sqrt(v_hat) + eps_);
    }
    // update params
    return {params - deltas, AdamOptimizer{beta1_, beta2_, eps_, std::move(m),
                                           std::move(v), step_ + 1}};
  }

 private:
  float beta1_, beta2_, eps_;
  ParameterGradients<M> m_;
  ParameterGradients<M> v_;
  size_t step_;
};

uchen::demo::DotGameReplay SelfPlay(uint32_t steps,
                                    const ModelParameters<Game::QModel>& par,
                                    int seed, float use_model) {
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

  std::uniform_real_distribution<> is_model{0.f, 1.f};

  for (size_t step = 0; step < steps; ++step) {
    size_t ind;
    if (is_model(gen) < use_model) {
      ind = dots_game.SuggestMove(par);
    } else {
      std::vector<int> good_indexes = dots_game.GetGoodAutoplayerIndexes();
      std::uniform_int_distribution<> dis(0, good_indexes.size() - 1);
      ind = good_indexes[dis(gen)];
    }
    dots_game.PlaceDot(ind, player);
    LOG(INFO) << "Step " << step << " Random index from good_indexes: " << ind
              << " player " << player
              << " score: " << dots_game.player_score(player);
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
    replay.RecordTurn(dots_game, step, ind, player);
    player = 3 - player;
  }
  LOG(INFO) << "Done, " << steps
            << " steps, player 1 score: " << dots_game.player_score(1)
            << ", player 2 score " << dots_game.player_score(2);
  return replay;
}

std::optional<std::ofstream> OpenFileForWrite(std::string_view filename,
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
                          std::ostream& stream, size_t* out_wrote) {
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
    const uchen::ModelParameters<Model>& parameters, std::ostream& stream,
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

template <typename Model>
std::optional<ModelParameters<Model>> ReadParameters(const Model* model,
                                                     std::ifstream& stream) {
  // allocate flat store
  auto store = uchen::NewFlatStore(model, 0.0f);
  std::span<float> data = store->data();

  // read all parameters at once
  stream.read(reinterpret_cast<char*>(data.data()),
              data.size() * sizeof(float));
  if (!stream) {
    return std::nullopt;
  }

  // wrap in ModelParameters
  return ModelParameters<Model>(model, std::move(store));
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

using ModelTraining =
    uchen::training::TrainingData<Game::QModel::input_t,
                                  uchen::learning::DeepQExpectation>;

uchen::ModelParameters<Game::QModel> TrainingLoop(
    const uchen::ModelParameters<Game::QModel>& params,
    const ModelTraining& training_data, const ModelTraining& verification,
    std::ostream& oss) {
  uchen::training::Training training(&Game::model, params,
                                     uchen::learning::DeepQLoss{},
                                     AdamOptimizer<Game::QModel>{});
  float loss = training.Loss(verification);
  LOG(INFO) << "Data size " << training_data.size() << " initial loss " << loss;
  for (size_t generation = 1; loss > 0.026; ++generation) {
    training = training.Generation(training_data, 0.0001);
    loss = training.Loss(verification);
    LOG(INFO) << absl::Substitute("Generation $0 loss $1", generation, loss);
    CHECK(WriteParameters(training.parameters(), oss,
                          Game::QModel::kLayerIndexes));
    if (generation > 50) {
      LOG(ERROR) << "Taking too long!";
    }
  }
  LOG(INFO) << "Training finished, loss " << training.Loss(verification);
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
    std::optional in_param = OpenFileForRead(absl::GetFlag(FLAGS_input_params));
    if (!in_param.has_value()) {
      return 1;
    }
    std::optional par = ReadParameters(&Game::model, *in_param);
    if (!par.has_value()) {
      LOG(FATAL) << "Unable to read parameters";
      return 1;
    }
    auto ofs = OpenFileForWrite(l.back(), absl::GetFlag(FLAGS_force));
    if (!ofs.has_value()) {
      return 1;
    }
    auto replay =
        SelfPlay(absl::GetFlag(FLAGS_steps), *par, absl::GetFlag(FLAGS_seed),
                 absl::GetFlag(FLAGS_model_play));
    if (!replay.Write(*ofs)) {
      return 1;
    }
    return 0;
  } else if (verb == "init_parameters") {
    auto ofs = OpenFileForWrite(absl::GetFlag(FLAGS_output_params),
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
    std::optional in_param = OpenFileForRead(absl::GetFlag(FLAGS_input_params));
    if (!in_param.has_value()) {
      return 1;
    }
    std::optional par = ReadParameters(&Game::model, *in_param);
    if (!par.has_value()) {
      LOG(FATAL) << "Unable to read parameters";
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
    auto [training, verification] =
        uchen::training::TrainingData<Game::QModel::input_t,
                                      uchen::learning::DeepQExpectation>(
            std::make_move_iterator(training_batch.begin()),
            std::make_move_iterator(training_batch.end()))
            .Shuffle()
            .Split(0.8f);

    std::optional out_params = OpenFileForWrite(
        absl::GetFlag(FLAGS_output_params), absl::GetFlag(FLAGS_force));
    if (!out_params.has_value()) {
      return 1;
    }
    uchen::ModelParameters params =
        TrainingLoop(*par, training, verification, *out_params);

    return 0;
  }
  std::cerr << "Unknown verb: " << verb;
  return 1;
}