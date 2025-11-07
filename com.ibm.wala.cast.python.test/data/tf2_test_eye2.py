import tensorflow as tf


def f(ab):
    pass


# Construct one identity matrix.
arg = tf.eye(2, None)
assert isinstance(arg, tf.Tensor)
assert arg.dtype == tf.float32
assert arg.shape == (2, 2)

f(arg)
