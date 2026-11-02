# The result of subscripting an array or a tensor with an index tuple that holds `...`, `None`
# (`tf.newaxis`) or a slice, with `None` alone, or with a list, is a new array or tensor; so is
# the result of a full integer index, a scalar of the array's dtype, modeled as a 0-d array
# (https://github.com/wala/ML/issues/400).
import numpy as np
import tensorflow as tf


def consume_ellipsis_newaxis(x):
    pass


def consume_ellipsis_none(x):
    pass


def consume_none_ellipsis(x):
    pass


def consume_slice_ellipsis_newaxis(x):
    pass


def consume_slice_none(x):
    pass


def consume_tensor_ellipsis_newaxis(x):
    pass


def consume_tensor_slice_ellipsis_newaxis(x):
    pass


def consume_none_alone(x):
    pass


def consume_integer_pair(x):
    pass


def consume_tensor_integer_pair(x):
    pass


def consume_fancy(x):
    pass


def consume_cast_scaled(x):
    pass


def consume_tensor_cast_scaled(x):
    pass


a = np.ones((3, 2), dtype=np.float32)

x1 = a[..., tf.newaxis]
assert x1.shape == (3, 2, 1) and x1.dtype == np.float32
consume_ellipsis_newaxis(x1)

x2 = a[..., None]
assert x2.shape == (3, 2, 1) and x2.dtype == np.float32
consume_ellipsis_none(x2)

x3 = a[None, ...]
assert x3.shape == (1, 3, 2) and x3.dtype == np.float32
consume_none_ellipsis(x3)

x4 = a[:2, ..., tf.newaxis]
assert x4.shape == (2, 2, 1) and x4.dtype == np.float32
consume_slice_ellipsis_newaxis(x4)

x5 = a[:, 0, None]
assert x5.shape == (3, 1) and x5.dtype == np.float32
consume_slice_none(x5)

x6 = a[None]
assert x6.shape == (1, 3, 2) and x6.dtype == np.float32
consume_none_alone(x6)

x7 = a[0, 1]
assert x7.shape == () and x7.dtype == np.float32
consume_integer_pair(x7)

x8 = a[[0, 2]]
assert x8.shape == (2, 2) and x8.dtype == np.float32
consume_fancy(x8)

t = tf.ones((3, 2))

x9 = t[..., tf.newaxis]
assert x9.shape == (3, 2, 1) and x9.dtype == tf.float32
consume_tensor_ellipsis_newaxis(x9)

x10 = t[:2, ..., tf.newaxis]
assert x10.shape == (2, 2, 1) and x10.dtype == tf.float32
consume_tensor_slice_ellipsis_newaxis(x10)

x11 = t[0, 1]
assert x11.shape == () and x11.dtype == tf.float32
consume_tensor_integer_pair(x11)

# A subscript's result used further: cast, then scaled.
x12 = a[..., None].astype(np.float32) / 255.0
assert x12.shape == (3, 2, 1) and x12.dtype == np.float32
consume_cast_scaled(x12)

x13 = tf.cast(t[..., tf.newaxis], tf.float32) / 255.0
assert x13.shape == (3, 2, 1) and x13.dtype == tf.float32
consume_tensor_cast_scaled(x13)
