import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(LayerNormalization, self).__init__()

    def call(self, x):
        return x
