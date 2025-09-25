#ifndef UCHEN_TEST_LIB_H
#define UCHEN_TEST_LIB_H

#include <vector>

#include <gtest/gtest.h>

namespace uchen::testing {

inline std::vector<float> RearrangeLinear(
    const std::vector<std::vector<float>>& data) {
  std::vector<float> result;
  size_t outputs = data.size();
  for (size_t i = 1; i < outputs; i++) {
    EXPECT_EQ(data[i - 1].size(), data[i].size()) << i;
  }
  result.resize(outputs * data[0].size());
  for (size_t i = 0; i < outputs; ++i) {
    result[i] = data[i].back();
    for (size_t j = 0; j < data[i].size() - 1; ++j) {
      result[(j + 1) * outputs + i] = data[i][j];
    }
  }
  return result;
}

}  // namespace uchen::testing

#endif  // UCHEN_TEST_LIB_H