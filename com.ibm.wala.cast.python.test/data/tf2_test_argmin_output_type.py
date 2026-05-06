import tensorflow as tf


def f(a):
    pass


# Counterpart of `tf2_test_argmax_output_type.py` for `tf.math.argmin`.
# `Argmin` extends `Argmax`, so the same `output_type` machinery applies.
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.math.argmin(x, axis=0, output_type=tf.int32)
assert isinstance(y, tf.Tensor)
assert y.dtype == tf.int32

f(y)
