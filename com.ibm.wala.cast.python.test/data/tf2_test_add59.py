import tensorflow as tf


def add(a, b):
    return a + b


arg1 = tf.ragged.constant([1, 2])
assert arg1.shape == (2,)
assert arg1.dtype == tf.int32

arg2 = tf.ragged.constant([2, 2])
assert arg2.shape == (2,)
assert arg2.dtype == tf.int32

c = add(arg1, arg2)
