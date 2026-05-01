import tensorflow as tf


def f(a):
    pass


arg = tf.eye(2, 3, [3, 2])

assert arg.shape == (3, 2, 2, 3)
assert arg.dtype == tf.float32

f(arg)
