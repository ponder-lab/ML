import tensorflow as tf


def f(a):
    pass


a = tf.sparse.eye(5, 2)
assert a.shape == (5, 2)
assert a.dtype == tf.float32

f(a)
