import tensorflow as tf


def f(a):
    pass


# `tf.clip_by_value(t, clip_value_min, clip_value_max)` is a pure
# passthrough — output shape and dtype both inherit from `t`.
t = tf.constant([1.0, 2.0, 3.0])
y = tf.clip_by_value(t, 1.5, 2.5)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
