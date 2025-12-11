import tensorflow as tf


def add(a, b):
    return a + b


arg1 = tf.fill([1, 2], 2.0)
assert arg1.shape == (1, 2)
assert arg1.dtype == tf.float32

arg2 = tf.fill([2, 2], 1.0)
assert arg2.shape == (2, 2)
assert arg2.dtype == tf.float32

c = add(arg1, arg2)
