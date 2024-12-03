"""
Python 2.7.16
TensorFlow 1.15.0
"""

MODEL_NAME = "toy_model_v1"

import os
import sys

cwd = os.getcwd()
if cwd in sys.path:
  sys.path.remove(cwd)
print(sys.path)

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

graph_file_path = "{}/origin/frozen_graph.origin".format(MODEL_NAME)

# create the input placeholder
X1 = tf.placeholder(tf.float32, shape=[None, 4], name="X1")
X2 = tf.placeholder(tf.float32, shape=[None, 4], name="X2")
# M2 = tf.constant(np.random.rand(3, 4), name="M2", dtype=tf.float32)
# M3 = tf.constant(np.random.rand(3, 4), name="M3", dtype=tf.float32)
# M4 = tf.constant(np.random.rand(3, 4), name="M4", dtype=tf.float32)
# M5 = tf.constant(np.random.rand(3, 4), name="M5", dtype=tf.float32)

I1 = tf.add(X1, X2, name="I1")
# I2 = tf.add(I1, M2, name="I2")
# I3 = tf.add(I2, M3, name="I3")
# I4 = tf.add(I3, M4, name="I4")
# I5 = tf.add(I4, M5, name="I5")

# compute output
output = tf.sigmoid(I1, name="Sigmoid")

# launch the graph in a session
with tf.Session() as sess:
  # # create the dictionary:
  d = {
    "X1:0": np.random.rand(3, 4),
    "X2:0": np.random.rand(3, 4)
  }

  # # feed it to placeholder a via the dict 
  # print(sess.run(C, feed_dict=d))
  print(sess.run(output, feed_dict=d))

  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  graph_pbtxt_file = graph_file_path + ".pbtxt"
  graph_writer = open(graph_pbtxt_file, "w")
  graph_writer.write(str(graph_def))
  graph_writer.close()
  print(graph_pbtxt_file)
  with tf.gfile.GFile(graph_file_path, 'wb') as f:
    data = graph_def.SerializeToString()
    f.write(data)
  print(graph_file_path)

# Create runstep
from google.protobuf import text_format
from bytedance.erdosidl.lagrange.common.graph_pb2 import GraphConfig

# Let's read our pbtxt file into a Graph protobuf
f = open("{}/origin/runstep.pbtxt".format(MODEL_NAME), "r")
runstep = text_format.Parse(f.read(), GraphConfig())

# Import the graph protobuf into our new graph.
with tf.gfile.GFile("{}/origin/runstep".format(MODEL_NAME), 'wb') as f:
  data = runstep.SerializeToString()
  f.write(data)
