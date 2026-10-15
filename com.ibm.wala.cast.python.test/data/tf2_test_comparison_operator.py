# Test for https://github.com/wala/ML/issues/1074.
import tensorflow as tf


def operator_sink(y):
    assert y.shape == ()
    assert y.dtype == tf.bool
    return y


def api_sink(y):
    assert y.shape == ()
    assert y.dtype == tf.bool
    return y


def f(x):
    operator_sink(tf.reduce_sum(x) > 0.0)
    api_sink(tf.greater(tf.reduce_sum(x), 0.0))


f(tf.ones((2, 2)))
