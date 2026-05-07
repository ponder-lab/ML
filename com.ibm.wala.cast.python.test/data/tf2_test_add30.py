import tensorflow as tf


def add(a, b):
    return a + b


arg1 = tf.fill([1, 2], 2)
assert arg1.shape == (1, 2)
assert arg1.dtype == tf.int32

c = add(arg1, tf.fill([2, 2], 1))
