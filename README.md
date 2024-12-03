## How it works
[./tf2stablehlo/tf2stablehlo.cc](./tf2stablehlo/tf2stablehlo.cc) converts a TensorFlow frozen graph to StableHLO.

[./compiler/compiler.cc](./compiler/compiler.cc) compile and execute the StableHLO.
<br/>

## Build
```Bash
$ ./build.sh
```
Artifacts will be copied to `./output` directory:
```Bash
.
├── bin
│   ├── compiler
│   └── tf2stablehlo
├── lib
│   └── libtensorflow_framework.so.2
├── run.sh
└── sample
    ├── frozen_graph.origin
    └── frozen_graph.origin.pbtxt
```

## Run
```Bash
$ cd output
$ ./run.sh 
2024-12-03 13:27:36.225054: I external/org_tensorflow/tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-12-03 13:27:36.264881: I external/org_tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.cc:234] Verbosity level too low to enable IR printing.
/data00/home/son.nguyen/workspace/openxla_dev/learnopenxla/output/sample/frozen_graph.mlir
/data00/home/son.nguyen/workspace/openxla_dev/learnopenxla/output/sample/frozen_graph.stablehlo

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1733232456.306332 4080408 service.cc:152] XLA service 0x5574c866fe20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1733232456.306361 4080408 service.cc:160]   StreamExecutor device (0): Host, Default Version
Loaded StableHLO program from sample/frozen_graph.stablehlo:
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "X1,X2", outputs = "Sigmoid:0"}} {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3x4xf32>
    %1 = stablehlo.logistic %0 : tensor<3x4xf32>
    return %1 : tensor<3x4xf32>
  }
}

2024-12-03 13:27:36.315472: I external/xla/xla/service/llvm_ir/llvm_command_line_options.cc:50] XLA (re)initializing LLVM with options fingerprint: 9451315669584224488
Computation inputs:
        x1:f32[3,4] {
  { 0, 0.0909090936, 0.181818187, 0.272727281 },
  { 0.363636374, 0.454545468, 0.545454562, 0.636363626 },
  { 0.727272749, 0.818181813, 0.909090936, 1 }
}
        x2:f32[3,4] {
  { 0, 0.0909090936, 0.181818187, 0.272727281 },
  { 0.363636374, 0.454545468, 0.545454562, 0.636363626 },
  { 0.727272749, 0.818181813, 0.909090936, 1 }
}
Computation output: f32[3,4] {
  { 0.5, 0.54532975, 0.589920402, 0.633080363 },
  { 0.674206495, 0.712814093, 0.748552859, 0.78120929 },
  { 0.810697, 0.83703959, 0.860347807, 0.880797 }
}
```
