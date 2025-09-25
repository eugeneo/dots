cc_binary(
    name = "attention",
    srcs = ["attention.benchmark.cc"],
    deps = [
        "//uchen/text:attention",
        "@google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "linear",
    srcs = ["linear.cc"],
    deps = [
        "//uchen:runtime",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:initialize",
        "@google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "matrix_math",
    srcs = ["matrix_math.benchmark.cc"],
    deps = [
        "//uchen:runtime",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:initialize",
        "@google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "matrixmul",
    srcs = ["matrixmul.benchmark.cc"],
    deps = [
        "//uchen/math:matrix",
        "//uchen/math:primitives",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:initialize",
        "@google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "rnn_names",
    srcs = ["rnn_names.cc"],
    deps = [
        "//datasets/names",
        "//experimental:rnn_names",
        "//uchen:runtime",
        "//uchen/training",
        "//uchen/training:rnn",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/status",
        "@google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "softmax",
    srcs = ["softmax.cc"],
    deps = [
        "//uchen:runtime",
        "//uchen:softmax",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:initialize",
        "@google_benchmark//:benchmark_main",
    ],
)
