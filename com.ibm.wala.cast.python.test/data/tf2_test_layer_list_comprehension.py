# Probe for the collection-dataflow family (wala/ML#618): layers built by a LIST COMPREHENSION
# (mirroring gpt-2's `self.decoder_layers = [DecoderLayer(...) for _ in range(num_layers)]`),
# iterated with `zip` against a `[None] * n` list, dispatch and their forward results type.
import tensorflow as tf


def consume(t):
    pass


class Inner(tf.keras.layers.Layer):
    def __init__(self):
        super(Inner, self).__init__()

    def call(self, inputs, past):
        h = tf.linalg.matmul(inputs, inputs)
        return h, past


class Outer(tf.keras.layers.Layer):
    def __init__(self):
        super(Outer, self).__init__()
        self.inner_layers = [Inner() for _ in range(3)]

    def call(self, inputs):
        pasts = [None] * 3
        hidden = inputs
        for inner_layer, past in zip(self.inner_layers, pasts):
            hidden, present = inner_layer(hidden, past)
        consume(hidden)
        return hidden


outer = Outer()
out = outer(tf.ones((4, 4)))
assert out.shape == (4, 4)
assert out.dtype == tf.float32
