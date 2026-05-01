import tensorflow as tf


def f(a):
    pass


# Construct one 2 x 3 "identity" matrix
arg = tf.eye(2, 3)
assert arg.shape == (2, 3)
assert arg.dtype == tf.float32

f(arg)
