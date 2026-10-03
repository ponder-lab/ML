import tensorflow as tf


def make():
    return tf.ones((2, 3))


class Maker:
    def __init__(self, n):
        self.n = n

    def make(self):
        return tf.ones((self.n, 4))


class Base:
    def __init__(self):
        self.t = tf.ones((2, 3))
