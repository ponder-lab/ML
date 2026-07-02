# Probe driver for wala/ML#618: the vendored `EmbeddingLayer`'s forward result in isolation.
import tensorflow as tf

from layers.embedding_layer import EmbeddingLayer


def consume(t):
    pass


emb = EmbeddingLayer(10, 8)
x = tf.constant([[1, 2, 3], [4, 5, 6]])
out = emb(x)
consume(out)
assert out.shape == (2, 3, 8)
assert out.dtype == tf.float32
