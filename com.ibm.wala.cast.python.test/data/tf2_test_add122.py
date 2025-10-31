import tensorflow as tf


def add(a, b):
    return a + b


arg1 = tf.random.normal([4], 0, 1, tf.float64)
assert arg1.shape == (4,)
assert arg1.dtype == tf.float64

c = add(arg1, tf.random.normal([4], 2, 1, tf.float64))
