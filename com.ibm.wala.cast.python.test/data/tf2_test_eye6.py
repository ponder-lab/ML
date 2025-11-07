import tensorflow as tf


def f(a):
    pass


batch = tf.constant([3, 2])
assert batch.shape == (2,)
assert batch.dtype == tf.int32

arg = tf.eye(2, 3, batch)

assert arg.shape == (3, 2, 2, 3)
assert arg.dtype == tf.float32

f(arg)
