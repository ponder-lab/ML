# Test for wala/ML#570: the result of a nested layer call whose inner `call` returns through an
# INHERITED method (`self.propagate(...)` defined on a same-module base class, mirroring
# `gcn_proj`'s `GraphConvolution(MessagePassing)` chain) carries the forward-result type.
from typing import NamedTuple

import tensorflow as tf


def consume(t):
    pass


class GNNInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: tf.Tensor


class MessagePassing(tf.keras.layers.Layer):
    def __init__(self):
        super(MessagePassing, self).__init__()

    def propagate(self, inputs):
        x = inputs.node_embeddings
        adj = inputs.adjacency_lists
        h = tf.linalg.matmul(adj, x)
        agg = tf.math.unsorted_segment_sum(h, tf.constant([0, 1, 2, 3]), 4)
        return agg


class Inner(MessagePassing):
    def __init__(self):
        super(Inner, self).__init__()

    def call(self, inputs):
        return self.propagate(inputs)


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
