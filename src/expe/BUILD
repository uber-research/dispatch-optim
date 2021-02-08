load("@python3_project_deps//:requirements.bzl", "all_requirements")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@brezel//rules/doe:gke.bzl", "doe_gke")


py_binary(
    name = "expe_var_mh",
    srcs = ["expe_var_mh.py"],
    deps = ["//src:dispatch-optim"],
)


py_binary(
    name = "expe_milp_mh",
    srcs = ["expe_milp_mh.py"],
    deps = ["//src:dispatch-optim"],
)


py_binary(
    name = "expe_mh_pertub",
    srcs = ["expe_mh_pertub.py"],
    deps = ["//src:dispatch-optim"],
)


py_binary(
    name = "expe_mh_parralel",
    srcs = ["expe_mh_parralel.py"],
    deps = ["//src:dispatch-optim"],
)


py_binary(
    name = "dummy_parr",
    srcs = ["dummy_parr.py"],
    deps = ["//src:dispatch-optim"],
)


py_binary(
    name = "expe_pool_routing",
    srcs = ["expe_pool_routing.py"],
    deps = ["//src:dispatch-optim"],
)

py_binary(
    name = "expe_pool_vs_tree",
    srcs = ["expe_pool_vs_tree.py"],
    deps = ["//src:dispatch-optim"],
)

py_binary(
    name = "expe_joint_vs_sequential",
    srcs = ["expe_joint_vs_sequential.py"],
    deps = ["//src:dispatch-optim"],
)

py3_image(
    name = "image-joint-seq",
    main = "expe_joint_vs_sequential.py",
    srcs = ["expe_joint_vs_sequential.py"],
    base = "@brezel//docker:python3_base",
    layers = ["//src:dispatch-optim"],
    imports = [".."],
)

py3_image(
    name = "image-tree-rs",
    main = "expe_pool_vs_tree.py",
    srcs = ["expe_pool_vs_tree.py"],
    base = "@brezel//docker:python3_base",
    layers = ["//src:dispatch-optim"],
    imports = [".."],
)

py3_image(
    name = "image-var-mh",
    main = "expe_var_mh.py",
    srcs = ["expe_var_mh.py"],
    base = "@brezel//docker:python3_base",
    layers = ["//src:dispatch-optim"],
    imports = [".."],
)

py3_image(
    name = "image-dummy-parr",
    main = "dummy_parr.py",
    srcs = ["dummy_parr.py"],
    base = "@brezel//docker:python3_base",
    layers = ["//src:dispatch-optim"],
    imports = [".."],
)

py3_image(
    name = "image-parr-mh",
    main = "expe_mh_parralel.py",
    srcs = ["expe_mh_parralel.py"],
    base = "@brezel//docker:python3_base",
    layers = ["//src:dispatch-optim"],
    imports = [".."],
)


py3_image(
    name = "image-milp-mh",
    main = "expe_milp_mh.py",
    srcs = ["expe_milp_mh.py"],
    base = "@brezel//docker:python3_base",
    layers = ["//src:dispatch-optim"],
    imports = [".."],
)



py3_image(
    name = "image-mh-pertub",
    main = "expe_mh_pertub.py",
    srcs = ["expe_mh_pertub.py"],
    base = "@brezel//docker:python3_base",
    layers = ["//src:dispatch-optim"],
    imports = [".."],
)


py3_image(
    name = "image-joint-pr",
    main = "expe_pool_routing.py",
    srcs = ["expe_pool_routing.py"],
    base = "@brezel//docker:python3_base",
    layers = ["//src:dispatch-optim"],
    imports = [".."],
)



doe_gke(
    name = "dispatch-optim-gke-milp-mh",
    image = {"eu.gcr.io/testing/dispatch-optim:latest": ":image-milp-mh"},
    bucket = "gs://data/experiments/dispatch-optim/output",
    matrix = ":params_milp_mh.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)


doe_gke(
    name = "dispatch-optim-gke-variance-mh-randstart",
    image = {"eu.gcr.io/testing/dispatch-optim:var": ":image-var-mh"},
    bucket = "gs://data/experiments/dispatch-optim/output",
    matrix = ":params_var_mh.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)

doe_gke(
    name = "dispatch-optim-gke-mh-pertub",
    image = {"eu.gcr.io/testing/dispatch-optim:var": ":image-mh-pertub"},
    bucket = "gs://data/experiments/dispatch-optim/output",
    matrix = ":params_mh_pertub.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)


doe_gke(
    name = "dispatch-optim-gke-mvp-mh",
    image = {"eu.gcr.io/testing/dispatch-optim:var": ":image-var-mh"},
    bucket = "gs://data/experiments/dispatch-optim/output_speed_test_secondb",
    matrix = ":params_mvp.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)


doe_gke(
    name = "dispatch-optim-gke-parr-mh",
    image = {"eu.gcr.io/testing/dispatch-optim:var": ":image-parr-mh"},
    bucket = "gs://data/experiments/dispatch-optim/output",
    matrix = ":params_parr.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)



doe_gke(
    name = "dispatch-optim-gke-parr-dummy",
    image = {"eu.gcr.io/testing/dispatch-optim:var": ":image-dummy-parr"},
    bucket = "gs://data/experiments/dispatch-optim/output",
    matrix = ":params_dummy.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)



doe_gke(
    name = "dispatch-optim-gke-joint-optim",
    image = {"eu.gcr.io/testing/dispatch-optim:var": ":image-joint-pr"},
    bucket = "gs://data/experiments/dispatch-optim/output_joint",
    matrix = ":params_joint.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)



doe_gke(
    name = "dispatch-optim-gke-rs-ts",
    image = {"eu.gcr.io/testing/dispatch-optim:var": ":image-tree-rs"},
    bucket = "gs://data/experiments/dispatch-optim/output_rs_ts",
    matrix = ":params_tree_rs.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)


doe_gke(
    name = "dispatch-optim-gke-joint-seq",
    image = {"eu.gcr.io/testing/dispatch-optim:var": ":image-joint-seq"},
    bucket = "gs://data/experiments/dispatch-optim/output_joint_seq",
    matrix = ":params_joint_seq.mat",
    output = "/tmp/results",
    nodepool = "pool-experiments",
    requests = {"cpu": "1.5"},
)
