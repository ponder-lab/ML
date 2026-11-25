# Test https://github.com/wala/ML/issues/1327: a call of `tf.keras.layers.Add` on a list of tensors
# reads as their sum, of the first input's type.
import tensorflow as tf


def consume_added(t):
    assert t.shape == (2, 3)


def consume_summed(t):
    assert t.shape == (2, 3)


class Residual(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)
        self.res_add = tf.keras.layers.Add()

    def call(self, inputs):
        return self.res_add([inputs, 0.5 * self.dense(inputs)])


x = tf.ones([2, 3])
consume_added(tf.keras.layers.Add()([x, x]))
consume_summed(Residual()(x))
