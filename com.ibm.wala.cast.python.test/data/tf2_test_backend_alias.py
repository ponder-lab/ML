# Probe for wala/ML#666: a dotted module import alias (`import tensorflow.keras.backend as K`)
# read inside a method resolves, so `K.cast(...)` types.
import tensorflow as tf
import tensorflow.keras.backend as K


def consume(t):
    pass


class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, inputs):
        h = K.cast(inputs, "float32")
        consume(h)
        return h


layer = MyLayer()
out = layer(tf.ones((4, 4)))
assert out.shape == (4, 4)
assert out.dtype == tf.float32
