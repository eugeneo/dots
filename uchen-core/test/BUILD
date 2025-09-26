cc_test(
    name = "arena_test",
    size = "small",
    srcs = ["arena.test.cc"],
    deps = [
        ":test_lib",
        "//uchen:runtime",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "layers_test",
    size = "small",
    srcs = ["layers.test.cc"],
    deps = [
        ":test_lib",
        "//uchen:runtime",
        "//uchen:softmax",
        "//uchen/math:matrix",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_test",
    size = "small",
    srcs = ["model.test.cc"],
    deps = [
        ":test_lib",
        "//uchen:runtime",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "parameters_test",
    size = "small",
    srcs = ["parameters.test.cc"],
    deps = [
        ":test_lib",
        "//uchen:runtime",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "rnn_test",
    size = "small",
    srcs = ["rnn.test.cc"],
    deps = [
        ":test_lib",
        "//uchen:rnn",
        "//uchen:runtime",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_lib",
    testonly = True,
    srcs = [],
    hdrs = ["test_lib.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@googletest//:gtest",
    ],
)
