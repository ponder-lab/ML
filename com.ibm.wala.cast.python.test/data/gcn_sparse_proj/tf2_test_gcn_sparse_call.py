import numpy as np
import tensorflow as tf

from deep_recommenders.keras.models.retrieval.gcn import GCN

# Driver for `GCN.call(self, features, adj, **kwargs)` from
# `LongmaoTeamTf/deep_recommenders` (`deep_recommenders/keras/models/retrieval/gcn.py`),
# a real-world graph-convolution layer, for tensor-type inference coverage. Unlike the
# other vendored layer methods, the `adj` parameter is a sparse adjacency
# (`tf.SparseTensor`): `GCN.call` branches on `isinstance(adj, tf.SparseTensor)` and uses
# `tf.sparse.sparse_dense_matmul`. This measures whether a sparse-tensor parameter is
# recovered. Mirrors `train_gcn_on_cora_keras.py`'s `GCN(32)(feats, g)` call site, where
# `g` is a `scipy.sparse` adjacency.
num_nodes = 4
num_features = 8

features = tf.constant(np.ones((num_nodes, num_features), dtype=np.float32))
adj = tf.SparseTensor(
    indices=[[0, 1], [1, 2], [2, 3], [3, 0]],
    values=tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32),
    dense_shape=[num_nodes, num_nodes],
)

model = GCN(units=16)
result = model(features, adj)

assert features.shape == (num_nodes, num_features)
assert features.dtype == tf.float32
assert isinstance(adj, tf.SparseTensor)
assert result.shape == (num_nodes, 16)
assert result.dtype == tf.float32
