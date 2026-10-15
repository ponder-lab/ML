# Test for https://github.com/wala/ML/issues/1074.
import numpy as np
import tensorflow as tf


def eq_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def ne_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def lt_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def le_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def gt_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def ge_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def scalar_left_sink(y):
    assert y.shape == ()
    assert y.dtype == tf.bool
    return y


def scalar_right_sink(y):
    assert y.shape == ()
    assert y.dtype == tf.bool
    return y


def variable_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def chained_sink(y):
    assert y.shape == ()
    assert y.dtype == tf.bool
    return y


def int_sink(y):
    assert isinstance(y, bool)
    return y


def len_sink(y):
    assert isinstance(y, bool)
    return y


def eq_none_sink(y):
    assert isinstance(y, bool)
    return y


def ne_none_sink(y):
    assert isinstance(y, bool)
    return y


def is_none_sink(y):
    assert isinstance(y, bool)
    return y


def is_not_none_sink(y):
    assert isinstance(y, bool)
    return y


def object_sink(y):
    assert isinstance(y, bool)
    return y


def dtype_sink(y):
    assert isinstance(y, bool)
    return y


def shape_sink(y):
    assert isinstance(y, bool)
    return y


def dim_sink(y):
    assert isinstance(y, bool)
    return y


def string_sink(y):
    assert isinstance(y, bool)
    return y


def layer_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def unmodeled_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def optional_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == tf.bool
    return y


def mixed_sink(y):
    assert isinstance(y, bool) or y.dtype == tf.bool
    return y


def numpy_sink(y):
    assert y.shape == (2, 3)
    assert y.dtype == np.bool_
    return y


def identity_sink(y):
    assert isinstance(y, bool)
    return y


def dynamic_shape_sink(y):
    assert y.shape == ()
    assert y.dtype == tf.bool
    return y


def element_shape_sink(y):
    assert isinstance(y, bool)
    return y


def element_dtype_sink(y):
    assert isinstance(y, bool)
    return y


def element_dim_sink(y):
    assert isinstance(y, bool)
    return y


class Box:
    pass


class Identity(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs


def optional(mask=None):
    if mask is not None:
        optional_sink(mask > 0.0)


def mixed(a):
    mixed_sink(a == a)


def element(a):
    element_shape_sink(a.shape == (3,))
    element_dtype_sink(a.dtype == tf.float32)
    element_dim_sink(a.shape[0] == 3)


def f(x, b, v, i, n, xs, mode):
    eq_sink(x == b)
    ne_sink(x != b)
    lt_sink(x < b)
    le_sink(x <= b)
    gt_sink(x > b)
    ge_sink(x >= b)
    s = tf.reduce_sum(x)
    scalar_left_sink(0.0 < s)
    scalar_right_sink(s > 0.0)
    variable_sink(v >= x)
    chained_sink(0.0 < s < 100.0)
    int_sink(i < n)
    len_sink(len(xs) > 0)
    eq_none_sink(x == None)
    ne_none_sink(x != None)
    is_none_sink(x is None)
    is_not_none_sink(x is not None)
    object_sink(Box() == Box())
    dtype_sink(x.dtype == tf.float32)
    shape_sink(x.shape == (2, 3))
    dim_sink(x.shape[0] == 2)
    string_sink(mode == "train")
    layer_sink(Identity()(x) > 0.0)
    unmodeled_sink(tf.math.bessel_i0e(x) > 0.0)
    optional()
    optional(x)
    mixed(Box())
    mixed(x)
    numpy_sink(np.ones((2, 3)) > 0)
    identity_sink(x is b)
    dynamic_shape_sink(tf.shape(x)[0] == 2)
    first, second = tf.unstack(x)
    element(first)


f(
    tf.ones((2, 3)),
    tf.zeros((3,)),
    tf.Variable(tf.ones((2, 3))),
    1,
    2,
    [1, 2],
    "train",
)
