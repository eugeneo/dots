package(default_visibility = ["//visibility:public"])

cc_library(
    name = "matrix",
    srcs = [],
    hdrs = [
        "concepts.h",
        "matrix.h",
        "matrix_core.h",
        "matrix_functional.h",
        "multiplication.h",
    ],
    deps = [
        ":primitives",
        "@abseil-cpp//absl/log:log",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "primitives",
    srcs = ["primitives.cc"],
    hdrs = ["primitives.h"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:log",
        "@highway//:algo",
        "@highway//:dot",
        "@highway//:hwy",
        "@highway//:math",
    ],
)

cc_library(
    name = "softmax",
    srcs = [],
    hdrs = ["softmax.h"],
    deps = [
        ":matrix",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:log",
        "@abseil-cpp//absl/strings:str_format",
    ],
)
