load("@python3_deps//:requirements.bzl", "requirement")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@python3_project_deps//:requirements.bzl", "all_requirements")

py_library(
    name = "app-dep",
    srcs = glob(["*.py"], exclude=["app.py"]),
    deps = all_requirements,
)


py_binary(
    name = "app",
    main = "app.py",
    srcs = ["app.py"],
    deps = [
        ":app-dep",

    ],
)