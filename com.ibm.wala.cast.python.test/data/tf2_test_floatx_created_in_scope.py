# Test https://github.com/wala/ML/issues/871: Keras reads floatx() when it creates a layer or a token,
# and later uses keep that value. A layer created between a set_floatx and its restore computes in the
# value set, and builds its weights in it, even when it is called after the restore.
import tensorflow as tf


def consume_output(x):
    assert x.dtype == tf.float64


def consume_weight(x):
    assert x.dtype == tf.float64


def consume_call_input(x):
    assert x.dtype == tf.float64


def consume_token_use(x):
    assert x.dtype == tf.float64


class Scale(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.w = self.add_weight(shape=(2, 3))

    def call(self, x):
        consume_call_input(x)
        consume_weight(self.w)
        return x


def make():
    tf.keras.backend.set_floatx("float64")
    dense = tf.keras.layers.Dense(3)
    scale = Scale()
    token = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx("float32")
    return dense, scale, token


dense, scale, token = make()
consume_output(dense(tf.ones([2, 2], tf.float64)))
scale(tf.ones([2, 2]))
consume_token_use(tf.ones([2, 2], dtype=token))
