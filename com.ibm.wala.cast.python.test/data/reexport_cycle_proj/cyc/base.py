import tensorflow as tf


class Base(tf.keras.layers.Layer):
    def call(self, x):
        return x * 2.0
