load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_project_deps//:requirements.bzl", project_requirements="requirement")

py_binary(
    name = "gen_figures",
    srcs = ["gen_figures.py"],
    deps = [
        ":illustrations",
    ],
)

py_library(
    name = "illustrations",
    srcs = ["illustrations.py"],
    deps = [
        requirement("numpy"),
        project_requirements("matplotlib"),
        "//src/electric:utils_electric",

    ],
)