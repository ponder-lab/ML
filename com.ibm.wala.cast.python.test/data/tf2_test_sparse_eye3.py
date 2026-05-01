import tensorflow as tf


def f(a):
    pass


a = tf.sparse.eye(5, None, tf.int32)
assert a.shape == (5, 5)
assert a.dtype == tf.int32

f(a)
