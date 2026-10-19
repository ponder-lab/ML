# Test for wala/ML#1120: a `Sequential` model's call passes each layer the `training` value it is
# called with, by keyword, so it binds to a layer's `training` parameter wherever that parameter is,
# and not to the parameter a layer declares after its input.
import tensorflow as tf


class Block(tf.keras.layers.Layer):
    def call(self, x, training=False):
        assert training is True
        return x


class Masked(tf.keras.layers.Layer):
    def call(self, inputs, mask=None, training=None):
        assert mask is None and training is True
        return inputs


model = tf.keras.Sequential([Block(), Masked(), tf.keras.layers.Dense(3)])
assert model(tf.ones([8, 4]), training=True).shape == (8, 3)
