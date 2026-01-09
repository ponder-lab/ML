import tensorflow as tf


def f(a):
    pass


# Construct a batch of 3 identity matrices, each 2 x 2.
# batch_identity[i, :, :] is a 2 x 2 identity matrix, i = 0, 1, 2.
arg = tf.eye(2, None, [3])
assert arg.shape == (3, 2, 2)
assert arg.dtype == tf.float32

f(arg)
