import numpy as np
import tensorflow as tf

from deep_recommenders.keras.models.retrieval.gcn import GCN


def consume(t):
    pass


# Inter-function-flow measurement: GCN.call is invoked twice, the second layer
# fed the first layer's OUTPUT (not a concrete driver), mirroring
# train_gcn_on_cora_keras.py's `x = GCN(32)(feats, g); GCN(num_classes)(x, g)`.
# `consume(hidden)` pins the first layer's return type, localizing whether the
# layer output is tensor-typed at all (and thus whether the downstream layer's
# `features` parameter can recover).
num_nodes = 4
num_features = 8

features = tf.constant(np.ones((num_nodes, num_features), dtype=np.float32))
adj = tf.SparseTensor(
    indices=[[0, 1], [1, 2], [2, 3], [3, 0]],
    values=tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32),
    dense_shape=[num_nodes, num_nodes],
)

gcn1 = GCN(units=16)
hidden = gcn1(features, adj)
consume(hidden)
gcn2 = GCN(units=8)
result = gcn2(hidden, adj)

assert hidden.shape == (num_nodes, 16)
assert hidden.dtype == tf.float32
assert result.shape == (num_nodes, 8)
