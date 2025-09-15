#include <emscripten/bind.h>

#include <iostream>
#include <string>

std::string concat_strings(const std::string &a, const std::string &b) {
  return a + b;
}

EMSCRIPTEN_BINDINGS(my_module) {
  emscripten::function("concat_strings", &concat_strings);
}

int main() {
  std::cout << "Hello, WebAssembly!" << std::endl;
  return 0;
}