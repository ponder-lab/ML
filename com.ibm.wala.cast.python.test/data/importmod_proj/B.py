# wala/ML#687 MRE: a sibling script module imported by A-variants.
import tensorflow as tf
import tensorflow.keras as keras


class Padding2D(keras.layers.Layer):
    def call(self, x):
        return tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="symmetric")
