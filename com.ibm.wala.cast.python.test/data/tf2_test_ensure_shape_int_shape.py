# Test `tf.ensure_shape`, whose result is its input with the declared static shape, and
# `tf.keras.backend.int_shape`, whose result is a Python tuple of the input's static extents and
# never a tensor.
import numpy as np
import tensorflow as tf


def consume_known(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_none_dim(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_refined(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_dynamic(t):
    assert t.shape.as_list() == [None, 3] and t.dtype == tf.float32


def consume_batch(t):
    assert t.shape.as_list() == [None, 3] and t.dtype == tf.float32


def consume_list_plus(t):
    assert t.shape == (1, 2, 3) and t.dtype == tf.float32


def consume_mixed_partner(t):
    assert t.shape == (2, 3) and t.dtype in (tf.float32, np.float64)


def consume_bool_list_plus(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def mixed_partner(c):
    partner = tf.ensure_shape(known, [2, 3]) if c else np.ones((2, 3), np.float32)
    consume_mixed_partner([1] + partner)


def consume_layer_plus(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_binop_plus(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


class Twice(tf.keras.layers.Layer):
    def call(self, x):
        return x * 2.0


def consume_user_layer_plus(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_int_shape(s):
    assert s == (2, 3)


def consume_int_shape_element(n):
    assert n == 2


def consume_zeros(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


known = tf.ones([2, 3])
consume_known(tf.ensure_shape(known, [2, 3]))
consume_none_dim(tf.ensure_shape(known, [None, 3]))
# `[1] + t` runs `t.__radd__`, which converts the list with `t`'s dtype.
consume_list_plus(tf.expand_dims([1] + tf.ensure_shape(known, [2, 3]), 0))
mixed_partner(True)
mixed_partner(False)
# `[1]` beside a layer call's result is converted with the result's float32 too.
consume_layer_plus([1] + tf.keras.layers.Dense(3)(tf.ones([2, 4])))
consume_binop_plus([1] + known * 2.0)
consume_user_layer_plus([1] + Twice()(known))
# `[True] + t` is converted with `t`'s dtype too.
consume_bool_list_plus([True] + tf.ensure_shape(known, [2, 3]))


# A Keras input's batch extent is `None`.
batch = tf.keras.Input(shape=(3,))
consume_batch(batch)
consume_refined(tf.ensure_shape(batch, [2, 3]))
consume_dynamic(tf.ensure_shape(batch, [None, 3]))

s = tf.keras.backend.int_shape(known)
consume_int_shape(s)
consume_int_shape_element(s[0])
consume_zeros(tf.zeros(tf.keras.backend.int_shape(known)))
