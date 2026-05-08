import tensorflow as tf


def f(a):
    pass


# Companion to `tf2_test_argmax_output_type.py` — exercises the
# *single-parameter sink, two call sites* shape: parameter `a` should
# union `x`'s and `y`'s tensor types across the two call sites. With
# `output_type=tf.int32`, the union is `{(2, 3) float32, ? int32}`.
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert x.shape == (2, 3)
assert x.dtype == tf.float32
y = tf.math.argmax(x, axis=0, output_type=tf.int32)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.int32

f(x)
f(y)
