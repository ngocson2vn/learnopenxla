load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "filegroup", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "tf2stablehlo",
    srcs = [
        "tf2stablehlo.cc"
    ],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo/cc:types",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo/cc:saved_model_import",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms"
    ]
)

cc_binary(
    name = "stablehlo_compiler",
    srcs = [
        "stablehlo_compiler.cc"
    ],
    deps = [
        "@local_xla//xla:error_spec",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla/stream_executor:executor_cache",
        "@local_xla//xla/stream_executor:stream_executor",
        "@local_xla//xla/stream_executor:cuda_platform",
        "@local_xla//xla/pjrt:local_device_state",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:pjrt_executable",
        "@local_xla//xla/pjrt:pjrt_stream_executor_client",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/service:stream_pool",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
)