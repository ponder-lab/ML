import tensorflow as tf


class Maker:
    def make(self, n=4):
        return tf.ones((n, 2))
