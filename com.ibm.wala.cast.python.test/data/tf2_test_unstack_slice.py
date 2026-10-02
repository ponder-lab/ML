# Test https://github.com/wala/ML/issues/993: a list a summary returns stands in for a value of
# unknown length, so a constant slice of it is not a slice of a known-length literal.
import tensorflow as tf


def first(t):
    assert t.shape == (2,)
    return t


def four(parts):
    a, b, c, d = parts
    first(a)
    return d


four(tf.unstack(tf.zeros([2, 8]), axis=-1)[0:4])
