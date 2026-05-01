import tensorflow as tf


def f(a):
    pass


a = tf.sparse.eye(5, 2, tf.int32)
assert a.shape == (5, 2)
assert a.dtype == tf.int32

f(a)
