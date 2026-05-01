import tensorflow as tf


def add(a, b):
    return tf.add(a, b)


a = tf.eye(2, 3)
assert a.shape == (2, 3)
assert a.dtype == tf.float32
b = tf.eye(2, 3)
assert b.shape == (2, 3)
assert b.dtype == tf.float32
c = add(a, b)
