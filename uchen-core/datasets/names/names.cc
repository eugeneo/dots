#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace uchen::datasets {

namespace {

constexpr int kNames = 100;
constexpr std::string_view kFilePath = "datasets/names/NationalNames.csv";

}  // namespace

std::vector<std::string> ReadNamesDb() {
  std::vector<std::string> nm;
  int boys = 0, girls = 0;
  std::ifstream file(std::filesystem::current_path() / kFilePath);
  while (nm.size() < kNames * 2) {
    std::string line;
    if (std::getline(file, line)) {
      std::string name, sex;
      std::istringstream iss(line);
      std::string token;
      int count = 0;
      while (std::getline(iss, token, ',')) {
        count++;
        if (count == 2) {
          name = token;
        } else if (count == 4) {
          sex = token;
        }
      }
      if (sex == "F" && girls < kNames) {
        girls++;
        nm.emplace_back("F" + name);
      } else if (sex == "M" && boys < kNames) {
        boys++;
        nm.emplace_back("M" + name);
      }
    }
  }
  file.close();
  std::seed_seq seed({1});
  std::mt19937 g(seed);
  std::shuffle(nm.begin(), nm.end(), g);
  return nm;
}

}  // namespace uchen::datasets