func.func @predict_online_0(%arg0: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<3x4>, input.from = "1:0", input.has_one_use = true} loc(unknown), %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<3x4>, input.from = "2:0", input.has_one_use = true} loc(unknown)) -> tensor<?x4xf32> attributes {SimpleFusion, _byted_af_group_idx = 0 : i64, _byted_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["Add:", "I1"])
  %1 = "tf.Sigmoid"(%0) {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["Sigmoid:", "Sigmoid"])
  return %1 : tensor<?x4xf32> loc(unknown)
} loc(unknown)
