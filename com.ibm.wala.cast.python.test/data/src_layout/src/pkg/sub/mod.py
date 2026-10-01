import tensorflow as tf


def consume(x):
    assert x.shape == (4, 5) and x.dtype == tf.float32


def take(x):
    consume(x)
    return x
