import tensorflow as tf


def f(a):
    pass


# Test tf.fill with keyword arguments
# dims=[2, 3], value=9
t1 = tf.fill(dims=[2, 3], value=9)

assert isinstance(t1, tf.Tensor)
assert t1.shape == (2, 3)
# 9 is an int, so dtype should be int32 (or whatever default int type is, usually int32 in TF2)
assert t1.dtype == tf.int32

f(t1)
