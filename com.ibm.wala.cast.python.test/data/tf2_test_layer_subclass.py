# Test for wala/ML#118: a user subclass of `tf.keras.layers.Layer` resolves its base class in the
# class hierarchy instead of falling back to `object`.
import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, inputs):
        return inputs


layer = MyLayer()
assert isinstance(layer, tf.keras.layers.Layer)
