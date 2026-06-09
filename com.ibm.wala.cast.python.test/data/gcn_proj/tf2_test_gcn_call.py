import tensorflow as tf
import numpy as np

from nlpgnn.models.GCN import GCNLayer

# Driver for `GCNLayer.call` from `kyzhouhzau/NLPGNN/nlpgnn/models/GCN.py`, a
# real-world graph-neural-network utility (a two-layer graph-convolution message-
# passing model), for tensor-type inference coverage. The `GCNLayer`,
# `GraphConvolution`, and `MessagePassing` modules under `nlpgnn/` are vendored
# verbatim from upstream; only this driver and the reachable-slice `utils.py` are
# bespoke. Exercises the multi-module import path the analyzer must follow.
num_nodes = 4
num_features = 8

node_embeddings = tf.constant(np.ones((num_nodes, num_features), dtype=np.float32))
# One edge type: an `[E, 2]` list of (source, target) node-index pairs.
adjacency_lists = [tf.constant([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=tf.int32)]

model = GCNLayer(hidden_dim=16, num_class=3)
result = model(node_embeddings, adjacency_lists, training=False)
assert node_embeddings.shape == (num_nodes, num_features)
assert node_embeddings.dtype == tf.float32
assert result.shape == (num_nodes, 3) and result.dtype == tf.float32
