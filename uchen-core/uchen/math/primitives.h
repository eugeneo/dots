#ifndef UCHEN_MATH_PRIMITIVES_H
#define UCHEN_MATH_PRIMITIVES_H

#include <cstdint>
#include <span>

namespace uchen::math {

float DotProduct(std::span<const float> a, std::span<const float> b);

// Multiplies matrix A stored in column-major format by vector B and writes
// the result to C.
// This is tied to the way linear layers are implemented in the Uchen ML.
void MatrixByVector(std::span<const float> a, std::span<const float> b,
                    std::span<float> out);

// Softmax that walks the matrix that is stored in column-major format.
void ColumnWiseSoftmax(std::span<const float> in, std::span<float> out,
                       uint32_t rows);

// Softmax that walks the matrix that is stored in row-major format.
void RowWiseSoftmax(std::span<const float> in, std::span<float> out,
                    uint32_t cols);

// Returns the number of SIMD lanes.
// Code outside this header should not use this function. It is only here for
// tests that need to test edge cases.
uint16_t GetLanesForTest();

}  // namespace uchen::math

#endif  // UCHEN_MATH_PRIMITIVES_H