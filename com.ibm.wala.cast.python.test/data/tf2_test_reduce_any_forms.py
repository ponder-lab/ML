# Test for https://github.com/wala/ML/issues/1076.
import tensorflow as tf


def math_any_sink(y):
    assert y.shape == ()
    assert y.dtype == tf.bool
    return y


def axis_sink(y):
    assert y.shape == (2,)
    assert y.dtype == tf.bool
    return y


def positional_axis_sink(y):
    assert y.shape == (2,)
    assert y.dtype == tf.bool
    return y


def negative_axis_sink(y):
    assert y.shape == (2,)
    assert y.dtype == tf.bool
    return y


def axis_list_sink(y):
    assert y.shape == ()
    assert y.dtype == tf.bool
    return y


def keepdims_sink(y):
    assert y.shape == (2, 1)
    assert y.dtype == tf.bool
    return y


def keepdims_all_sink(y):
    assert y.shape == (1, 1)
    assert y.dtype == tf.bool
    return y


def float64_nan_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def f(x):
    m = tf.math.is_nan(x)
    math_any_sink(tf.math.reduce_any(m))
    axis_sink(tf.reduce_any(m, axis=1))
    positional_axis_sink(tf.reduce_any(m, 1))
    negative_axis_sink(tf.reduce_any(m, axis=-1))
    axis_list_sink(tf.reduce_any(m, axis=[0, 1]))
    keepdims_sink(tf.reduce_any(m, axis=1, keepdims=True))
    keepdims_all_sink(tf.reduce_any(m, keepdims=True))
    float64_nan_sink(tf.math.is_nan(tf.cast(x, tf.float64)))


f(tf.ones((2, 3)))
