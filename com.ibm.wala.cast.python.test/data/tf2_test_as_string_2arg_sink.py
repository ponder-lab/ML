# Sibling of `tf2_test_as_string.py` with a single 2-arg sink `f(y, x)`. Covers the
# multi-tensor-sink shape (the same shape that triggers wala/ML#495 in the dataset-loader
# case). For `as_string` on a `tf.constant`, the analyzer classifies both `y` (string dtype)
# and `x` (float32) precisely; the companion JUnit test asserts those types directly.
import tensorflow as tf


def f(a, b):
    pass


x = tf.constant([1.0, 2.0, 3.0])
assert isinstance(x, tf.Tensor)
assert x.shape == (3,)
assert x.dtype == tf.float32
y = tf.strings.as_string(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.string
f(y, x)
