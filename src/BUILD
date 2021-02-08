load("@python3_project_deps//:requirements.bzl", "all_requirements")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@brezel//rules/doe:gke.bzl", "doe_gke")

py_library(
    name = 'dispatch-optim',
    srcs = glob(["*.py"]),
    deps = all_requirements,
    visibility = ["//visibility:public"],
)

py_library(
    name = 'utils',
    srcs = ["utils.py",
    ],
    deps = all_requirements,
    imports = ["."],
    visibility = ["//visibility:public"],
)
