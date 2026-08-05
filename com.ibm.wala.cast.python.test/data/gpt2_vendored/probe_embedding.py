# Probe driver for wala/ML#618: the vendored `EmbeddingLayer`'s forward result in isolation.
import tensorflow as tf

from layers.embedding_layer import EmbeddingLayer


def consume(t):
    pass


emb = EmbeddingLayer(10, 8)
x = tf.constant([[1, 2, 3], [4, 5, 6]])
out = emb(x)
# Observed by running this file under `python3.10` (wala/ML#808). Asserted before the sink so the
# check runs whether or not the sink is reached, and so the sink's parameter carries no extra reads.
assert out.shape == (2, 3, 8) and out.dtype == tf.float32
consume(out)
