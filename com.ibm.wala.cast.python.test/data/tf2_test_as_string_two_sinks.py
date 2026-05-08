# Sibling of `tf2_test_as_string.py` with two 1-arg sinks (`f(y)` and `g(x)`). Covers the
# split-sink shape (related to wala/ML#495 in the dataset-loader case). For `as_string` on
# a `tf.constant`, the analyzer classifies both sinks' params precisely (`y` as string,
# `x` as float32); the companion JUnit tests assert those types directly.
import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
assert isinstance(x, tf.Tensor)
assert x.shape == (3,)
assert x.dtype == tf.float32
y = tf.strings.as_string(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.string
f(y)
g(x)
