#include "src/replay.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <span>
#include <utility>

#include "src/deepq_loss.h"
#include "src/game.h"

namespace uchen::demo {
namespace {
constexpr std::string_view kDotReplaysMark = "uchen-demo-dots\n";

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

bool WritePlayerLog(
    std::ostream& out,
    std::span<const DotGameReplay::SelfPlayTurnRecord> records) {
  std::array<char, sizeof(size_t) + 1> buffer;
  size_t size = records.size();
  std::copy_n(reinterpret_cast<char*>(&size), sizeof(size_t), buffer.begin());
  buffer.back() = '\n';
  out.write(buffer.data(), buffer.size());
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

void FillTensor(std::span<float> tensor, std::span<const uint32_t> input,
                uint8_t channel) {
  for (size_t index : input) {
    size_t c = index % 64;
    size_t r = index / 64;
    tensor[((c * 64) + r) * 4 + channel] = 1.f;
  }
}

Game::QModel::input_t EncodeAsTensor(
    const DotGameReplay::SelfPlayTurnRecord& record) {
  auto store = uchen::memory::ArrayStore<
      float, Game::QModel::input_t::elements>::NewInstance(0.f);
  std::span span = store->data();
  FillTensor(span, record.dots_our, 0);
  FillTensor(span, record.dots_our, 1);
  FillTensor(span, record.dots_our, 2);
  FillTensor(span, record.dots_our, 3);
  return Game::QModel::input_t{span, std::move(store)};
}

}  // namespace

// static
std::optional<DotGameReplay> DotGameReplay::Load(std::istream& is) {
  DotGameReplay replay;
  // Read and validate magic header
  std::string header;
  header.resize(kDotReplaysMark.size());
  is.read(header.data(), header.size());
  if (!is || header != kDotReplaysMark) {
    return replay;  // return empty on invalid header
  }

  auto read_vector = [&is](std::vector<uint32_t>& out) -> bool {
    uint32_t sz = 0;
    is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    if (!is) return false;
    out.resize(sz);
    if (sz > 0) {
      is.read(reinterpret_cast<char*>(out.data()), sz * sizeof(uint32_t));
      if (!is) return false;
    }
    return true;
  };

  auto read_player_log =
      [&is, &read_vector](std::vector<SelfPlayTurnRecord>& out) -> bool {
    // Read count (sizeof(size_t)) + one trailing '\n' char
    size_t count = 0;
    is.read(reinterpret_cast<char*>(&count), sizeof(count));
    char nl = '\0';
    is.read(&nl, 1);
    if (!is) return false;
    // Optional: validate newline marker
    // if (nl != '\n') return false;

    out.clear();
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      SelfPlayTurnRecord rec{};
      is.read(reinterpret_cast<char*>(&rec.step), sizeof(rec.step));
      is.read(reinterpret_cast<char*>(&rec.move), sizeof(rec.move));
      is.read(reinterpret_cast<char*>(&rec.score_our), sizeof(rec.score_our));
      is.read(reinterpret_cast<char*>(&rec.score_opponent),
              sizeof(rec.score_opponent));
      if (!is) return false;

      if (!read_vector(rec.dots_our)) return false;
      if (!read_vector(rec.dots_opponent)) return false;
      if (!read_vector(rec.captured_our)) return false;
      if (!read_vector(rec.captured_opponent)) return false;

      out.emplace_back(std::move(rec));
    }
    return true;
  };

  if (!read_player_log(replay.replays_[0]) ||
      !read_player_log(replay.replays_[1])) {
    return std::nullopt;
  }
  return replay;
}

void DotGameReplay::RecordTurn(const Game& game, int step, uint32_t move,
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
  replays_[player - 1].emplace_back(std::move(result));
}

bool DotGameReplay::Write(std::ostream& ostream) const {
  ostream << kDotReplaysMark;
  return WritePlayerLog(ostream, replays_[0]) &&
         WritePlayerLog(ostream, replays_[1]);
}

void BellmanRewards(auto begin, auto end, float gamma) {
  float reward = 0;
  for (auto it = begin; it != end; ++it) {
    it->second.bellman_target += reward * gamma;
    reward = it->second.bellman_target;
  }
}

void UpdateReplays(std::span<const DotGameReplay::SelfPlayTurnRecord> replays,
                   auto inserter) {
  float previous_score = 0;
  for (const auto& replay : replays) {
    auto result = std::pair(EncodeAsTensor(replay),
                          learning::DeepQExpectation{.action = replay.move});
    float score = replay.score_our * 10.f - replay.score_opponent;
    result.second.bellman_target = score - previous_score;
    previous_score = score;
    *(inserter++) = std::move(result);
  }
}

std::vector<std::pair<Game::QModel::input_t, learning::DeepQExpectation>>
DotGameReplay::ToTrainingSet(float gamma) const {
  std::vector<std::pair<Game::QModel::input_t, learning::DeepQExpectation>>
      result;
  UpdateReplays(replays_[0], std::back_inserter(result));
  size_t player1_records = result.size();
  UpdateReplays(replays_[1], std::back_inserter(result));
  BellmanRewards(result.rbegin(), result.rbegin() + player1_records, gamma);
  BellmanRewards(result.rbegin() + player1_records, result.rend(), gamma);
  return result;
}

bool operator==(const DotGameReplay& a, const DotGameReplay& b) {
  for (size_t p = 0; p < a.replays_.size(); ++p) {
    const auto& ar = a.replays_[p];
    const auto& br = b.replays_[p];
    if (ar.size() != br.size()) return false;
    for (size_t i = 0; i < ar.size(); ++i) {
      const auto& x = ar[i];
      const auto& y = br[i];
      if (x.step != y.step || x.move != y.move || x.score_our != y.score_our ||
          x.score_opponent != y.score_opponent) {
        return false;
      }
      if (x.dots_our != y.dots_our || x.dots_opponent != y.dots_opponent ||
          x.captured_our != y.captured_our ||
          x.captured_opponent != y.captured_opponent) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace uchen::demo