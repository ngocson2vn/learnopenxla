#loc = loc(unknown)
#loc1 = loc("Placeholder:")
#loc2 = loc("X1")
#loc3 = loc("X2")
#loc4 = loc("Sigmoid:")
#loc5 = loc("Sigmoid")
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func.func @main() -> tensor<?x4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "Sigmoid:0"}} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island {
        %1 = "tf.Placeholder"() {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = "", shape = #tf_type.shape<?x4>} : () -> tensor<?x4xf32> loc(#loc6)
        tf_executor.yield %1 : tensor<?x4xf32> loc(#loc6)
      } {_byted_af_op_idx = "1"} loc(#loc6)
      %outputs_0, %control_1 = tf_executor.island {
        %1 = "tf.Placeholder"() {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = "", shape = #tf_type.shape<?x4>} : () -> tensor<?x4xf32> loc(#loc7)
        tf_executor.yield %1 : tensor<?x4xf32> loc(#loc7)
      } {_byted_af_op_idx = "2"} loc(#loc7)
      %outputs_2, %control_3 = tf_executor.island {
        %1 = "tf.FusedCwise"(%outputs, %outputs_0) {_symbolic_output_shapes = [#tf_type.shape<3x4>], metadata = "predict_online_0"} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(#loc8)
        tf_executor.yield %1 : tensor<?x4xf32> loc(#loc8)
      } {_byted_af_group_idx = 0 : i64, _byted_af_op_idx = "3"} loc(#loc8)
      tf_executor.fetch %outputs_2 : tensor<?x4xf32> {_byted_af_op_idx = "4"} loc(#loc)
    } loc(#loc)
    return %0 : tensor<?x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc6 = loc(fused["Placeholder:", "X1"])
#loc7 = loc(fused["Placeholder:", "X2"])
#loc8 = loc(fused["Sigmoid:", "Sigmoid"])

