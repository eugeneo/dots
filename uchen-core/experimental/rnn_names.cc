#include "experimental/rnn_names.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace uchen::experimental {

std::vector<std::pair<const internal::Input, const char>>
PrepareTrainingDataSet(std::span<const std::string> names) {
  std::vector<std::pair<const internal::Input, const char>> dataset;
  for (std::string_view name : names) {
    for (size_t i = 1; i < name.length(); ++i) {
      char c = std::tolower(name[i]);
      DCHECK_GE(c, 'a');
      DCHECK_LE(c, 'z');
      dataset.emplace_back(internal::Input(name.substr(0, i)), c);
    }
    dataset.emplace_back(internal::Input(name), 'z' + 1);
  }
  return dataset;
}

}  // namespace uchen::experimental

namespace std {

std::ostream& operator<<(std::ostream& os,
                         uchen::experimental::internal::Input /* input */) {
  os << "(Input)";
  return os;
}

}  // namespace std
