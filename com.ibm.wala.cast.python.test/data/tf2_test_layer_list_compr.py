# Miniature of NLPGNN's `GAAELayer` indexed sub-layer dispatch (wala/ML#661 shape 3):
# a plain list of sublayers built by a list comprehension in `build`, dispatched through a
# dynamic subscript in `call`. `Inner.call` returns a distinctly-shaped tensor so the sub-layer
# forward result is observable at the sink: a masked dispatch failure would leave the input's
# (2, 3) shape (carried by the loop phi) rather than the sub-layer's (6, 1).
import tensorflow as tf


class Inner(tf.keras.layers.Layer):
    def call(self, inputs, training=True):
        return tf.ones((6, 1))


class Outer(tf.keras.layers.Layer):
    def __init__(self, num_layers=2, **kwargs):
        super(Outer, self).__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        self.sub_layers = [Inner() for _ in range(self.num_layers)]

    def call(self, inputs, training=True):
        out = inputs
        for i in range(self.num_layers):
            out = self.sub_layers[i](out, training)
        return out


def consume(t):
    pass


layer = Outer()
x = tf.ones((2, 3))
out = layer(x)

assert out.shape == (6, 1)
assert out.dtype == tf.float32

consume(out)
