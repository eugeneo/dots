#include <iostream>
#include <string>

#include <emscripten/bind.h>

#include "absl/strings/str_cat.h"

namespace test {

int add(int a, int b);
std::string simd();

} // namespace test

std::string concat_strings(const std::string &a, const std::string &b) {
  return absl::StrCat(a, " ", test::add(2, 3), " ", b, " ", test::simd());
}

EMSCRIPTEN_BINDINGS(my_module) {
  emscripten::function("concat_strings", &concat_strings);
}

int main() { return 0; }