import tensorflow as tf


class Maker:
    def make(self, n=5):
        return tf.ones((n, 3))
