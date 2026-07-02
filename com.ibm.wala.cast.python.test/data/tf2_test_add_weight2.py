# Probe for wala/ML#667: `add_weight`'s `shape` and `dtype` arguments are consumed, so the
# created weight itself types precisely. The dtype here is the `tf.float32` module constant;
# the string form is covered by tf2_test_add_weight.py.
import tensorflow as tf


def consume(t):
    pass


class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[4, 4], dtype=tf.float32, initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        consume(self.kernel)
        return inputs


layer = MyLayer()
out = layer(tf.ones((4, 4)))
assert layer.kernel.shape == (4, 4)
assert layer.kernel.dtype == tf.float32
