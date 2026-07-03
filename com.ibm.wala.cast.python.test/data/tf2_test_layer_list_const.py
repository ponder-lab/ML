# Miniature of NLPGNN's `GAAELayer` indexed sub-layer dispatch (wala/ML#661 shape 3):
# a plain list of sublayers populated by `append` in `build`, dispatched through a
# dynamic subscript in `call`.
import tensorflow as tf


class Inner(tf.keras.layers.Layer):
    def call(self, inputs, training=True):
        return inputs


class Outer(tf.keras.layers.Layer):
    def __init__(self, num_layers=2, **kwargs):
        super(Outer, self).__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        self.sub_layers = []
        for _ in range(self.num_layers):
            self.sub_layers.append(Inner())

    def call(self, inputs, training=True):
        out = inputs
        for i in range(self.num_layers):
            out = self.sub_layers[0](out, training)
        return out


def consume(t):
    pass


layer = Outer()
x = tf.ones((2, 3))
out = layer(x)
consume(out)

assert out.shape == (2, 3)
assert out.dtype == tf.float32
