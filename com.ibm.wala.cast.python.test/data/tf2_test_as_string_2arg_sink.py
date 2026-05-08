# Sibling of `tf2_test_as_string.py` with a single 2-arg sink `f(y, x)` exercising the
# wala/ML#495 multi-tensor-sink-collapse pattern. With both `y` (the as_string output) and
# `x` (the input) flowing into one sink, dataset-loader-style fallback paths in the analyzer
# collapse classification on both. The companion JUnit test captures the currently-observed
# (broken) result with a TODO referencing #495.
import tensorflow as tf


def f(a, b):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y = tf.strings.as_string(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.string
f(y, x)
