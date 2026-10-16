# Test for https://github.com/wala/ML/issues/1084.
import tensorflow as tf


def type_sink(y):
    assert isinstance(y, bool)
    return y


def ndim_sink(y):
    assert isinstance(y, bool)
    return y


def as_list_sink(y):
    assert isinstance(y, bool)
    return y


def dict_sink(y):
    assert isinstance(y, bool)
    return y


def element_sink(y):
    assert y.shape == (3,)
    assert y.dtype == tf.bool
    return y


def zeros_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def list_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def empty():
    return {}


def f(x):
    type_sink(type(x) == tf.Tensor)
    ndim_sink(x.ndim == 2)
    as_list_sink(x.shape.as_list()[1:] == [3])
    dict_sink(empty() == {})
    element_sink(x[0] > 0.0)
    zeros_sink(tf.zeros(x.shape) == 0.0)
    list_sink(x == [1.0, 2.0, 3.0])


f(tf.ones((2, 3)))
