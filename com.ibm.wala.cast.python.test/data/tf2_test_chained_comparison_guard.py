# Test for https://github.com/wala/ML/issues/1079.
import tensorflow as tf


def consume_chained(y):
    assert y.shape == (2,)
    assert y.dtype == tf.float32


def consume_split(y):
    assert y.shape == (2,)
    assert y.dtype == tf.float32


class Window:
    def __init__(self, m):
        self.m = m

    def chained(self, x):
        if 0 < self.m < 3:
            y = tf.reshape(x, [-1, 6])
        else:
            y = tf.reduce_mean(x, 1)
        consume_chained(y)

    def split(self, x):
        if 0 < self.m and self.m < 3:
            y = tf.reshape(x, [-1, 6])
        else:
            y = tf.reduce_mean(x, 1)
        consume_split(y)


Window(5).chained(tf.ones((2, 6)))
Window(5).split(tf.ones((2, 6)))
