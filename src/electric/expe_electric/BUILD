load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@brezel//rules/doe:gke.bzl", "doe_gke")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_project_deps//:requirements.bzl", project_requirements = "requirement")


py_binary(
    name = "benchmarch_vns",
    srcs = ["benchmarch_vns.py"],
    deps = [
            "//src/electric:utils_electric",
            "//src/electric:optim_electric",
            "//src:utils",
            requirement("click"),

          ],
)

py3_image(
    name = "image_benchmarch_vns",
    main = "benchmarch_vns.py",
    srcs = ["benchmarch_vns.py"],
    base = "@brezel//docker:python3_base",
    layers = [
              "//src:dispatch-optim",
              "//src/electric:utils_electric",
              "//src/electric:optim_electric",
              "//src:utils",
              requirement("click"),
            ],
    imports = [".."],
)

doe_gke(
    name = "bench-vns",
    image = {"eu.gcr.io/testing/dispatch-optim:latest": ":image_benchmarch_vns"},
    matrix = ":params_bench_vns.mat",
    gcs_upload = {"/tmp/results": "gs://data/experiments/dispatch-optim/output_electric/benchmark_vns"},
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)
