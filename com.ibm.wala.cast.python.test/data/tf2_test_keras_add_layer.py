# Test https://github.com/wala/ML/issues/1327: a call of `tf.keras.layers.Add` on a list of tensors
# reads as their sum, of the first input's type. Keras broadcasts the inputs and casts floating ones
# to the layer's compute dtype, which the first input's type misses (https://github.com/wala/ML/issues/1329).
import tensorflow as tf


def consume_added(t):
    assert t.shape == (2, 3)


def consume_summed(t):
    assert t.shape == (2, 3)


def consume_broadcast(t):
    assert t.shape == (2, 3)


def consume_autocast(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


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
consume_broadcast(tf.keras.layers.Add()([tf.ones([2, 1]), x]))
x64 = tf.ones([2, 3], tf.float64)
consume_autocast(tf.keras.layers.Add()([x64, x64]))
