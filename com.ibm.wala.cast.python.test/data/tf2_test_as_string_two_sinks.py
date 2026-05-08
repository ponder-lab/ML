# Sibling of `tf2_test_as_string.py` with two 1-arg sinks (`f(y)` and `g(x)`) exercising
# the wala/ML#495 multi-tensor-sink-collapse pattern. Even split across two separate sinks,
# adding a second tensor arg to the analyzer's per-sink view drops tensor classification
# on both. The companion JUnit tests capture the currently-observed (broken) result with a
# TODO referencing #495.
import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y = tf.strings.as_string(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.string
f(y)
g(x)
