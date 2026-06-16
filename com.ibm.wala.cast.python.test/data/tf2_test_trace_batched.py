import tensorflow as tf


def f(x, y):
    pass


# Batched counterpart of `tf2_test_trace.py`: `tf.linalg.trace` collapses the
# last two dimensions (the trace of each matrix in the batch), so a (3, 2, 2)
# input yields a (3,) output that inherits the input dtype.
x = tf.ones([3, 2, 2])
assert x.shape == (3, 2, 2)
assert x.dtype == tf.float32
y = tf.linalg.trace(x)
assert y.shape == (3,)
assert y.dtype == tf.float32

f(x, y)
