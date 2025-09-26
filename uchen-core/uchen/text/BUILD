cc_library(
    name = "attention",
    srcs = [],
    hdrs = [
        "attention.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":embeddings",
        "//uchen:runtime",
        "//uchen/math:matrix",
        "//uchen/math:softmax",
    ],
)

cc_library(
    name = "embeddings",
    srcs = [],
    hdrs = [
        "embeddings.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//uchen:runtime",
    ],
)

cc_library(
    name = "embeddings_training",
    srcs = [],
    hdrs = [
        "embeddings_training.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":embeddings",
        "//uchen:runtime",
    ],
)
