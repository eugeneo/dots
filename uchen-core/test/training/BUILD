load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "backprop_test",
    size = "small",
    srcs = ["backprop.test.cc"],
    deps = [
        "//test:test_lib",
        "//uchen:runtime",
        "//uchen/training",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "grad_e2e_test",
    size = "small",
    srcs = ["grad.e2e.test.cc"],
    deps = [
        "//uchen/training",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "kaiming_he_test",
    size = "small",
    srcs = ["kaiming_he.test.cc"],
    deps = [
        "//uchen:rnn",
        "//uchen:runtime",
        "//uchen/training:kaiminghe",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "loss_test",
    size = "small",
    srcs = ["loss.test.cc"],
    deps = [
        "//test:test_lib",
        "//uchen:runtime",
        "//uchen/training",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_gradients_test",
    size = "small",
    srcs = ["model_gradients.test.cc"],
    deps = [
        "//test:test_lib",
        "//uchen:runtime",
        "//uchen/training",
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
        "//test:test_lib",
        "//uchen:runtime",
        "//uchen/training",
        "//uchen/training:rnn",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "training_test",
    size = "small",
    srcs = ["training.test.cc"],
    deps = [
        "//test:test_lib",
        "//uchen/training",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest_main",
    ],
)
