load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "rnn",
    srcs = [],
    hdrs = [
        "rnn.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":runtime",
    ],
)

cc_library(
    name = "runtime",
    srcs = [],
    hdrs = [
        "inferrence_context.h",
        "layer_traits.h",
        "layers.h",
        "linear.h",
        "memory.h",
        "model.h",
        "parameters.h",
        "vector.h",
    ],
    copts = ["-std=c++20"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//uchen/math:matrix",
        "@abseil-cpp//absl/log:check",
    ],
)

cc_library(
    name = "softmax",
    srcs = [],
    hdrs = [
        "softmax.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":runtime",
        "//uchen/math:softmax",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    copts = [
        "-std=c++20",
        "-fno-exceptions",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:log",
    ],
)
