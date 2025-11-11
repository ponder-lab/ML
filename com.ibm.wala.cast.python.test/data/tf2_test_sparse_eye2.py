import tensorflow as tf


def f(a):
    pass


a = tf.sparse.eye(5, None, tf.float32)
assert a.shape == (5, 5)
assert a.dtype == tf.float32

f(a)
