import tensorflow as tf


def f(a):
    pass


# Test tf.fill with mixed arguments (positional dims, keyword value)
t1 = tf.fill([2, 3], value=9.0)

assert isinstance(t1, tf.Tensor)
assert t1.shape == (2, 3)
assert t1.dtype == tf.float32

f(t1)
