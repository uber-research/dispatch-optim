##
# Bazel rules from Brezel
##
local_repository(
    name = "brezel",
    path = "./third_party/brezel"
)

load("@brezel//rules:defaults.bzl", project_defaults="brezel_defaults")
project_defaults(
    gcp_project = "my-gcp-project",
    gcp_cluster = "my-gcp-cluster",
    gcp_bucket  = "gs://my-gcp-bucket",
    gcp_registry = "gcr.io/my-gcp-registry",
    srcs=["@brezel//config/infra/vars:gke.bzl"],
)

load("@brezel//third_party:deps.bzl", brzl_ext_libs="third_party_repositories")
brzl_ext_libs()

load("@brezel//third_party:rules.bzl", brzl_rules="rule_repositories")
brzl_rules()

load("@brezel//third_party:rules_deps.bzl", brzl_deps="all_indirect_repositories")
brzl_deps()

load("@brezel//third_party/toolchains:prepare_toolchains.bzl", brzl_prepare="prepare_all_toolchains")
brzl_prepare()

load("@brezel//third_party/toolchains:toolchains.bzl", brzl_toolchains="setup_all_toolchains")
brzl_toolchains()

##
# Project specific rules
##
load("@rules_python_external//:defs.bzl", "pip_install")
pip_install(
    name = "python3_project_deps",
    requirements = "//third_party/pip:requirements.txt",
)
