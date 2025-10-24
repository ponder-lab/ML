import tensorflow as tf


def add(a, b):
    return a + b


arg = tf.zeros_like([1, 2], tf.float32)
assert arg.shape == (2,)
assert arg.dtype == tf.float32

c = add(arg, tf.zeros_like([2, 2], tf.float32))
