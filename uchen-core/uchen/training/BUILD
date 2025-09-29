load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "training",
    srcs = [],
    hdrs = [
        "backprop.h",
        "loss.h",
        "model_gradients.h",
        "parameter_gradients.h",
        "training.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//uchen:runtime",
        "//uchen:softmax",
        "//uchen/input:text",
        "//uchen/math:matrix",
        "@abseil-cpp//absl/log",
    ],
)

cc_library(
    name = "rnn",
    srcs = [],
    hdrs = ["rnn.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":training",
        "//uchen:rnn",
        "//uchen:runtime",
    ],
)

cc_library(
    name = "kaiminghe",
    srcs = [],
    hdrs = ["kaiming_he.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//uchen:rnn",
        "//uchen:runtime",
    ],
)
