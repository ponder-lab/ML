import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(EmbeddingLayer, self).__init__()

    def call(self, x, mode="embedding"):
        return x


class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(PositionEmbeddingLayer, self).__init__()

    def call(self, x, start=0):
        return x
