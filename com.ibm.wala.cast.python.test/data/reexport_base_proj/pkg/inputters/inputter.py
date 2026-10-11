import tensorflow as tf


class Inputter(tf.keras.layers.Layer):
    def call(self, x):
        return x * 3.0


class ExampleInputter(Inputter):
    def __init__(self, features_inputter):
        super().__init__()
        self.features_inputter = features_inputter


class WordEmbedder(Inputter):
    def call(self, x):
        return x * 2.0
