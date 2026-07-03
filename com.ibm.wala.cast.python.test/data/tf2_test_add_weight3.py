# Pins wala/ML#672: `add_weight` without a `dtype` argument follows Keras's documented default
# and types float32 (the layer variable dtype under the default global policy). Completes the
# dtype-form trio: string ("float32", tf2_test_add_weight.py), module constant (tf.float32,
# tf2_test_add_weight2.py), and absent (this fixture).
import tensorflow as tf


def consume(t):
    pass


class Linear(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[2, 4])

    def call(self, x):
        consume(self.w)
        return tf.matmul(x, self.w)


layer = Linear()
out = layer(tf.ones((3, 2)))

assert tuple(layer.w.shape) == (2, 4)
assert layer.w.dtype == tf.float32
