#include <cstdint>
#include <string>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"

using namespace emscripten;

namespace test {

int add(int a, int b);
std::string simd();

} // namespace test

std::string concat_strings(const std::string &a, const std::string &b) {
  return absl::StrCat(a, " ", test::add(2, 3), " ", b, " ", test::simd());
}

class Game {
public:
  Game(int h, int w) : vec_(h * w, 0) {
    field_ = val(typed_memory_view(vec_.size(), vec_.data()));
  }

  emscripten::val GetField() const { return field_; }

  void GameTurn(size_t index, uint8_t player_id) {
    vec_[index] = player_id;
    LOG(INFO) << absl::Substitute("Player $0 placed to $1",
                                  static_cast<int>(player_id), index);
  }

private:
  absl::InlinedVector<uint8_t, 64 * 64> vec_;
  emscripten::val field_;
};

EMSCRIPTEN_BINDINGS(dots_logic) {
  emscripten::function("concat_strings", &concat_strings);
  class_<Game>("Game")
      .constructor<int, int>()
      .function("GetField", &Game::GetField)
      .function("GameTurn", &Game::GameTurn);
}

int main() { return 0; }