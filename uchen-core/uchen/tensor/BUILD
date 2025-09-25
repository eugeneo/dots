load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensor",
    srcs = [
        "float_tensor.cc",
    ],
    hdrs = [
        "float_tensor.h",
        "function.h",
        "special_tensors.h",
    ],
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@highway//:hwy",
    ],
)
