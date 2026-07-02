# Probe for wala/ML#661 shape 3: an INDEXED sub-layer call (`self.container[i](x)`, mirroring
# NLPGNN's `GAAELayer.encoder`'s `self.encoder_layers[i](GNNInput(...), ...)`).
import tensorflow as tf


def consume(t):
    pass


class Inner(tf.keras.layers.Layer):
    def __init__(self):
        super(Inner, self).__init__()

    def call(self, inputs):
        return tf.linalg.matmul(inputs, inputs)


class Outer(tf.keras.layers.Layer):
    def __init__(self):
        super(Outer, self).__init__()
        self.inner_layers = [Inner() for _ in range(3)]

    def call(self, inputs):
        hidden = inputs
        for i in range(3):
            hidden = self.inner_layers[i](hidden)
        consume(hidden)
        return hidden


outer = Outer()
out = outer(tf.ones((4, 4)))
assert out.shape == (4, 4)
assert out.dtype == tf.float32
