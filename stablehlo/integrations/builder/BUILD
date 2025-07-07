load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/llvm/llvm-project/mlir:tblgen.bzl", "gentbl_cc_library")
load("//third_party/tensorflow:tensorflow.google.bzl", "get_compatible_with_portable")

package(
    default_applicable_licenses = ["//third_party/stablehlo:license"],  # copybara:comment
    default_visibility = ["//learning/brain/mlir:stablehlo_friends"],
    licenses = ["notice"],
)

#####
# [WIP] TableGen
#####

cc_binary(
    name = "mlir_builder_tblgen",
    srcs = ["MlirBuilderTblgen.cpp"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        "//third_party/absl/container:flat_hash_set",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/llvm:TableGen",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/llvm/llvm-project/mlir:TableGen",
    ],
)

####
## MlirBuilder base class
####

cc_library(
    name = "mlir_builder",
    srcs = ["MlirBuilder.cpp"],
    hdrs = ["MlirBuilder.h"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        ":attr_type_builder_util",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
    ],
)

####
## Attr / Type Builder Helpers
####

cc_library(
    name = "attr_type_builder_util",
    srcs = ["AttrTypeBuilderUtil.cpp"],
    hdrs = ["AttrTypeBuilderUtil.h"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
    ],
)

cc_test(
    name = "attr_type_builder_util_test",
    srcs = ["AttrTypeBuilderUtilTest.cpp"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        ":attr_type_builder_util",
        "//testing/base/public:gunit_main",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
    ],
)

#####
## Dialect-specific builders
####

gentbl_cc_library(
    name = "chlo_builder_inc",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ChloBuilder.h.inc": ["-gen-builder-decls"],
        "ChloBuilder.cpp.inc": ["-gen-builder-defs"],
        "ChloBuilder.md": ["-gen-builder-docs"],
    },
    tblgen = ":mlir_builder_tblgen",
    td_file = "//third_party/stablehlo:stablehlo/dialect/ChloOps.td",
    deps = [
        "//third_party/llvm/llvm-project/mlir:InferTypeOpInterfaceTdFiles",
        "//third_party/llvm/llvm-project/mlir:OpBaseTdFiles",
        "//third_party/llvm/llvm-project/mlir:SideEffectInterfacesTdFiles",
        "//third_party/stablehlo:chlo_ops_td_files",
    ],
)

cc_library(
    name = "chlo_builder",
    srcs = ["ChloBuilder.cpp"],
    hdrs = ["ChloBuilder.h"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        ":chlo_builder_inc",
        ":mlir_builder",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/stablehlo:chlo_ops",
    ],
)

gentbl_cc_library(
    name = "func_builder_inc",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "FuncBuilder.h.inc": ["-gen-builder-decls"],
        "FuncBuilder.cpp.inc": ["-gen-builder-defs"],
        "FuncBuilder.md": ["-gen-builder-docs"],
    },
    tblgen = ":mlir_builder_tblgen",
    td_file = "//third_party/llvm/llvm-project/mlir:include/mlir/Dialect/Func/IR/FuncOps.td",
    deps = [
        "//third_party/llvm/llvm-project/mlir:FuncTdFiles",
        "//third_party/llvm/llvm-project/mlir:InferTypeOpInterfaceTdFiles",
        "//third_party/llvm/llvm-project/mlir:OpBaseTdFiles",
        "//third_party/llvm/llvm-project/mlir:SideEffectInterfacesTdFiles",
    ],
)

cc_library(
    name = "func_builder",
    srcs = ["FuncBuilder.cpp"],
    hdrs = ["FuncBuilder.h"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        ":func_builder_inc",
        ":mlir_builder",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
    ],
)

gentbl_cc_library(
    name = "stablehlo_builder_inc",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "StablehloBuilder.h.inc": ["-gen-builder-decls"],
        "StablehloBuilder.cpp.inc": ["-gen-builder-defs"],
        "StablehloBuilder.md": ["-gen-builder-docs"],
    },
    tblgen = ":mlir_builder_tblgen",
    td_file = "//third_party/stablehlo:stablehlo/dialect/StablehloOps.td",
    deps = [
        "//third_party/llvm/llvm-project/mlir:InferTypeOpInterfaceTdFiles",
        "//third_party/llvm/llvm-project/mlir:OpBaseTdFiles",
        "//third_party/llvm/llvm-project/mlir:ShapeOpsTdFiles",
        "//third_party/llvm/llvm-project/mlir:SideEffectInterfacesTdFiles",
        "//third_party/stablehlo:base_td_files",
        "//third_party/stablehlo:stablehlo_ops_td_filegroup",
    ],
)

cc_library(
    name = "stablehlo_builder",
    srcs = ["StablehloBuilder.cpp"],
    hdrs = ["StablehloBuilder.h"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        ":attr_type_builder_util",
        ":mlir_builder",
        ":stablehlo_builder_inc",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:InferTypeOpInterface",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/stablehlo:stablehlo_ops",
        "//third_party/stablehlo:stablehlo_type_inference",
    ],
)

cc_test(
    name = "stablehlo_builder_test",
    srcs = ["StablehloBuilderTest.cpp"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        ":attr_type_builder_util",
        ":chlo_builder",
        ":func_builder",
        ":mlir_builder",
        ":stablehlo_builder",
        "//testing/base/public:gunit_main",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/stablehlo:register",
        "//third_party/stablehlo:stablehlo_ops",
    ],
)

cc_test(
    name = "mlir_builder_test",
    srcs = ["MlirBuilderTest.cpp"],
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        ":attr_type_builder_util",
        ":chlo_builder",
        ":func_builder",
        ":mlir_builder",
        ":stablehlo_builder",
        "//testing/base/public:gunit_main",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/llvm/llvm-project/mlir:TosaDialect",
        "//third_party/stablehlo:register",
        "//third_party/stablehlo:stablehlo_ops",
    ],
)
