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
        "@xla//xla:error_spec",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla/client:local_client",
        "@xla//xla/client:client_library",
        "@xla//xla/service:service",
        "@xla//xla/service:hlo_runner",
        "@xla//xla/pjrt:local_device_state",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt:tracked_device_buffer",
        "@xla//xla/pjrt/c:pjrt_c_api_cpu",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:path",
    ],
)
