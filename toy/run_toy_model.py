############################################################################################################
# Customize logging
############################################################################################################
import logging
import coloredlogs

# install a handler on the root logger
coloredlogs.install(
    level=logging.DEBUG,
    fmt="%(levelname)s %(message)s"
)

logging.debug("This is a debug message")
logging.warning("This is a warning message")
############################################################################################################

import os
import numpy as np
import tensorflow.compat.v1 as tf

from bytedance.tensorsharp.util import tensorflow_utils
tensorflow_utils.load_all_custom_ops(hardware="gpu")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", required=True, help="path to model")
args = parser.parse_args()
optimized_frozen_graph_path = args.model + "/optimize/optimized_graph_predict_online.pb"

print("optimized_frozen_graph_path: {}".format(optimized_frozen_graph_path))


# create the feed dictionary:
X1 = np.random.rand(3, 4)
X2 = np.random.rand(3, 4)
print("X1:\n", X1)
print("X2:\n", X2)
feeds = {
  "X1:0": X1,
  "X2:0": X2,
}

fetches = ["Sigmoid:0"]

# Import graph
with tf.gfile.GFile(optimized_frozen_graph_path, 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name='')
print("Successfully imported graph")

# launch the graph in a session
with tf.Session() as sess:
  # feed it to placeholder a via the dict
  print("{}:\n{}".format(fetches[0], sess.run(fetches=fetches, feed_dict=feeds)))
