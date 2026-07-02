# Cross-module base for wala/ML#570/#571: `propagate` is inherited across a module boundary.
import tensorflow as tf


class MessagePassing(tf.keras.layers.Layer):
    def __init__(self):
        super(MessagePassing, self).__init__()

    def propagate(self, inputs):
        x = inputs.node_embeddings
        adj = inputs.adjacency_lists
        h = tf.linalg.matmul(adj, x)
        agg = tf.math.unsorted_segment_sum(h, tf.constant([0, 1, 2, 3]), 4)
        return agg
