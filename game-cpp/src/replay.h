#ifndef SRC_REPLAY_H
#define SRC_REPLAY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

#include "src/deepq_loss.h"
#include "src/game.h"

namespace uchen::demo {

class DotGameReplay {
 public:
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
          "S:$7 M:$0 P1:$1 P2:$2 {$3}{$4}{$5}{$6}", record.move,
          record.score_our, record.score_opponent,
          absl::StrJoin(record.dots_our, ","),
          absl::StrJoin(record.dots_opponent, ","),
          absl::StrJoin(record.captured_our, ","),
          absl::StrJoin(record.captured_opponent, ","), record.step));
    }
  };

  static std::optional<DotGameReplay> Load(std::istream& is);

  void RecordTurn(const Game& game, int step, uint32_t move, uint32_t player);
  bool Write(std::ostream& ostream) const;

  size_t turns() const { return replays_[0].size() + replays_[1].size(); }

  std::vector<std::pair<Game::QModel::input_t, learning::DeepQExpectation>>
  ToTrainingSet(float gamma) const;

  friend bool operator==(const DotGameReplay& a, const DotGameReplay& b);

 private:
  std::array<std::vector<SelfPlayTurnRecord>, 2> replays_;
};

}  // namespace uchen::demo

#endif  // SRC_REPLAY_H