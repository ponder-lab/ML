# Test for https://github.com/wala/ML/issues/1076.
import tensorflow as tf


def nan_sink(x):
    assert x.shape == (2, 2)
    assert x.dtype == tf.bool
    return x


def any_sink(x):
    assert x.shape == ()
    assert x.dtype == tf.bool
    return x


def all_sink(x):
    assert x.shape == ()
    assert x.dtype == tf.bool
    return x


def f(x):
    nan_sink(tf.math.is_nan(x))
    any_sink(tf.reduce_any(tf.math.is_nan(x)))
    all_sink(tf.reduce_all(tf.equal(x, x)))


f(tf.ones((2, 2)))
