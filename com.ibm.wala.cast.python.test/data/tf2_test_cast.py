import tensorflow as tf


def f(a):
    pass


# `tf.cast(x, dtype)`: shape inherits from `x`; dtype is the explicit
# `dtype` argument. Here `x` is float32 (3,) and we cast to int32, so the
# result is (3,) int32.
x = tf.constant([1.5, 2.7, 3.1])
y = tf.cast(x, dtype=tf.int32)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.int32
f(y)
