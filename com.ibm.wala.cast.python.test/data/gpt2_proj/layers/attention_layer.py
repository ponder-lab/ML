import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MultiHeadAttention, self).__init__()

    def call(self, x, mask=None, past_layer=None, training=True):
        return x, None
