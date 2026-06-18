import tensorflow as tf
import numpy as np

from nlpgnn.models.GAT import GATLayer

# Driver for `GATLayer.call` from `kyzhouhzau/NLPGNN/nlpgnn/models/GAT.py`, a
# real-world graph-neural-network utility (a two-layer graph-attention message-
# passing model), for tensor-type inference coverage. The `GATLayer`,
# `GraphAttentionConvolution`, and `MessagePassing` modules under `nlpgnn/` are
# vendored verbatim from upstream; only this driver and the reachable-slice
# `utils.py` are bespoke. Parallels `gcn_proj/tf2_test_gcn_call.py` for the
# attention variant.
num_nodes = 4
num_features = 8

node_embeddings = tf.constant(np.ones((num_nodes, num_features), dtype=np.float32))
# One edge type: an `[E, 2]` list of (source, target) node-index pairs.
adjacency_lists = [tf.constant([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=tf.int32)]

num_class = 3
model = GATLayer(hidden_dim=8, num_class=num_class, heads=8)
result = model(node_embeddings, adjacency_lists, training=False)
assert node_embeddings.shape == (num_nodes, num_features)
assert node_embeddings.dtype == tf.float32
assert result.shape == (num_nodes, num_class) and result.dtype == tf.float32
