cc_test(
    name = "attention_test",
    size = "small",
    srcs = ["attention.test.cc"],
    copts = [
        "-std=c++20",
    ],
    deps = [
        "//uchen:runtime",
        "//uchen/text:attention",
        "//uchen/text:embeddings",
        "//uchen/text:embeddings_training",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "embeddings_test",
    size = "small",
    srcs = ["embeddings.test.cc"],
    copts = [
        "-std=c++20",
    ],
    deps = [
        "//uchen:runtime",
        "//uchen/text:embeddings",
        "//uchen/text:embeddings_training",
        "//uchen/training",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@googletest//:gtest",
    ],
)
