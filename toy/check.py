from byted_operators.utils.graph_def_utils import GraphDefViewer

base = "toy_model_v1/predict_online/tf.opt.fake.pb"
graph = GraphDefViewer.load(base)
print(graph._graph_def)

compute_op = 0
for node in graph.nodes:
    if node.op in ["Placeholder", "Const"]:
        continue
    compute_op += 1

print(compute_op)
