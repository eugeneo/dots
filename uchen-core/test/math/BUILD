load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "matrix_test",
    size = "small",
    srcs = ["matrix.test.cc"],
    deps = [
        "//uchen/math:matrix",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "matrix_math_test",
    size = "small",
    srcs = ["matrix_math.test.cc"],
    deps = [
        "//uchen/math:matrix",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "primitives_test",
    size = "small",
    srcs = ["primitives.test.cc"],
    deps = [
        "//uchen/math:matrix",
        "//uchen/math:primitives",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "softmax_test",
    size = "small",
    srcs = ["softmax.test.cc"],
    deps = [
        "//uchen/math:matrix",
        "//uchen/math:softmax",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest",
    ],
)
