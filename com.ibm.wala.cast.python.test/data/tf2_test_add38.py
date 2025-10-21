import tensorflow as tf


def add(a, b):
    return a + b


arg1 = tf.range(3, 18, 3)
assert arg1.shape == (5,)
assert arg1.dtype == tf.int32

arg2 = tf.range(5)
assert arg2.shape == (5,)
assert arg2.dtype == tf.int32

c = add(arg1, arg2)
