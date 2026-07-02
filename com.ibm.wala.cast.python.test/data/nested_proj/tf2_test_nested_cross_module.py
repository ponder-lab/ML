# Test for wala/ML#570: the result of a nested layer call whose inner `call` returns through a
# method inherited from a CROSS-MODULE base (mirroring `gcn_proj`'s
# `GraphConvolution(MessagePassing)` with `MessagePassing` in another file) carries the
# forward-result type.
from typing import NamedTuple

import tensorflow as tf

from inner import Inner


def consume(t):
    pass


class GNNInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: tf.Tensor


class Outer(tf.keras.layers.Layer):
    def __init__(self):
        super(Outer, self).__init__()
        self.inner = Inner()

    def call(self, node_embeddings, adjacency_lists):
        x = self.inner(GNNInput(node_embeddings, adjacency_lists))
        consume(x)
        return x


outer = Outer()
out = outer(tf.ones((4, 8)), tf.ones((4, 4)))
assert out.shape == (4, 8)
assert out.dtype == tf.float32
