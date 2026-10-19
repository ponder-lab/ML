import tensorflow as tf


def consume_taken(x):
    assert x.shape == (4, 5)
    assert x.dtype == tf.float32


def take(x):
    consume_taken(x)
    return x
