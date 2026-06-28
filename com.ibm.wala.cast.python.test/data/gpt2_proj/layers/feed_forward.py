import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(FeedForward, self).__init__()

    def call(self, x, training=True):
        return x
