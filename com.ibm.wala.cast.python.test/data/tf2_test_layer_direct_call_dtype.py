# Test for wala/ML#1114: applying a layer casts a floating-point input to the layer's compute dtype
# before `call` runs, and invoking `call` directly does not.
import tensorflow as tf


def applied_inside(t):
    assert t.dtype == tf.float32 and t.shape == (2,)
    return t


def direct_inside(t):
    assert t.dtype == tf.float64 and t.shape == (2,)
    return t


class Applied(tf.keras.layers.Layer):

    def call(self, inputs):
        return applied_inside(inputs)


class Direct(tf.keras.layers.Layer):

    def call(self, inputs):
        return direct_inside(inputs)


Applied()(tf.ones([2], tf.float64))
Direct().call(tf.ones([2], tf.float64))
