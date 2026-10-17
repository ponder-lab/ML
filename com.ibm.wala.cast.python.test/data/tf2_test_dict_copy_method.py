# Test for wala/ML#1082: a dictionary's `copy()` returns a new dictionary holding the same entries,
# so a function receiving the copy reads the original's values, and a write to the copy does not
# reach the original.
import tensorflow as tf


def consume_features(f):
    assert f.shape == (3,) and f.dtype == tf.float32
    return f


def consume_original(f):
    assert f.shape == (3,) and f.dtype == tf.float32
    return f


def make_features(t, features=None):
    consume_features(features["a"])
    return t


def caller(t, features):
    return make_features(t, features=features.copy())


caller(tf.ones((2,)), {"a": tf.ones((3,))})

original = {"a": tf.ones((3,))}
copied = original.copy()
copied["a"] = tf.zeros((4,))
consume_original(original["a"])
