# Test https://github.com/wala/ML/issues/1308: `np.expand_dims`, `np.squeeze` and `np.clip` each
# return a new array whose type comes from the argument's.

import time

import numpy as np
import tensorflow as tf


def call(self, x):
    return x + 1


def expanded(x):
    assert x.shape == (1, 2, 4) and x.dtype == np.float32


def expanded_at_the_end(x):
    assert x.shape == (2, 4, 1) and x.dtype == np.float32


def expanded_by_keyword(x):
    assert x.shape == (2, 1, 4) and x.dtype == np.float32


def expanded_by_axes(x):
    assert x.shape == (1, 1, 2, 4) and x.dtype == np.float32


def expanded_list(x):
    assert x.shape == (1, 3) and x.dtype == np.int64


def expanded_number(x):
    assert x.shape == (1,) and x.dtype == np.float64


def expanded_boolean(x):
    assert x.shape == (1,) and x.dtype == np.bool_


def expanded_true(x):
    assert x.shape == (1,) and x.dtype == np.bool_


def variable_of_expanded(x):
    assert x.shape == (1, 2, 4) and x.dtype == tf.float32


def constant_of_expanded(x):
    assert x.shape == (1, 2, 4) and x.dtype == tf.float32


def squeezed(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def squeezed_by_axis(x):
    assert x.shape == (2, 1) and x.dtype == np.float32


def squeezed_by_keyword(x):
    assert x.shape == (1, 2) and x.dtype == np.float32


def squeezed_list(x):
    assert x.shape == (2,) and x.dtype == np.float64


def variable_of_squeezed(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def clipped(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def clipped_from_below(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def clipped_by_keywords(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def clipped_by_negative(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def clipped_by_folded(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def clipped_complex(x):
    assert x.shape == (2, 2) and x.dtype == np.complex64


def clipped_doubles(x):
    assert x.shape == (2, 3) and x.dtype == np.float64


def clipped_doubles_by_floats(x):
    assert x.shape == (2, 3) and x.dtype == np.float64


def clipped_integers(x):
    assert x.shape == (2, 2) and x.dtype == np.int32


def clipped_integers_by_floats(x):
    assert x.shape == (2, 2) and x.dtype == np.float64


def clipped_bytes(x):
    assert x.shape == (2, 2) and x.dtype == np.uint8


def clipped_bytes_by_negative(x):
    # The dtype is the narrowest one that holds the array's values and `-1` under NumPy's
    # value-based casting, and the call raises from NumPy 2 on.
    assert x.shape == (2, 2) and x.dtype != np.uint8


def clipped_by_wide_integer(x):
    # Double precision under NumPy's value-based casting, and single precision from NumPy 2 on.
    assert x.shape == (2, 4)


def clipped_by_wide_negative(x):
    # Double precision under NumPy's value-based casting, and single precision from NumPy 2 on.
    assert x.shape == (2, 4)


def clipped_by_wide_float(x):
    # Double precision under NumPy's value-based casting, and the call overflows from NumPy 2 on.
    assert x.shape == (2, 4)


def clipped_bytes_by_wide(x):
    # The dtype is the narrowest one that holds the array's values and `300` under NumPy's
    # value-based casting, and the call raises from NumPy 2 on.
    assert x.shape == (2, 2) and x.dtype != np.uint8


def clipped_integers_by_wide(x):
    # A 64-bit integer under NumPy's value-based casting, and the call raises from NumPy 2 on.
    assert x.shape == (2, 2) and x.dtype != np.int32


def clipped_integers_by_wide_negative(x):
    # A 64-bit integer under NumPy's value-based casting, and the call raises from NumPy 2 on.
    assert x.shape == (2, 2) and x.dtype != np.int32


def clipped_by_array(x):
    assert x.shape == (3, 2, 4) and x.dtype == np.float64


def clipped_by_upper_array(x):
    assert x.shape == (3, 2, 4) and x.dtype == np.float64


def clipped_by_unread(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def clipped_by_booleans(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def clipped_row(x):
    assert x.shape == (4,) and x.dtype == np.float32


def clipped_list(x):
    assert x.shape == (3,) and x.dtype == np.int64


def clipped_tuple(x):
    assert x.shape == (3,) and x.dtype == np.float64


def clipped_number(x):
    assert x.shape == () and x.dtype == np.int64


def variable_of_clipped(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


boxes = np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)
ids = np.array([[1, 2], [3, 4]], dtype=np.int32)
pixels = np.array([[0, 128], [255, 7]], dtype=np.uint8)
lows = np.zeros((3, 1, 1), dtype=np.float64)
highs = np.ones((3, 1, 1), dtype=np.float64)
spectrum = np.ones((2, 2), dtype=np.complex64)
doubles = np.ones((2, 3), dtype=np.float64)

expanded(np.expand_dims(boxes, 0))
expanded_at_the_end(np.expand_dims(boxes, -1))
expanded_by_keyword(np.expand_dims(a=boxes, axis=1))
expanded_by_axes(np.expand_dims(boxes, (0, 1)))
expanded_list(np.expand_dims([1, 2, 3], 0))
expanded_number(np.expand_dims(float(boxes.shape[0]), 0))
expanded_boolean(np.expand_dims(bool(boxes.shape[0]), 0))
expanded_true(np.expand_dims(True, 0))
variable_of_expanded(tf.Variable(np.expand_dims(boxes, 0)))
constant_of_expanded(tf.constant(np.expand_dims(boxes, 0)))

squeezed(np.squeeze(np.zeros((1, 2, 4), dtype=np.float32)))
squeezed_by_axis(np.squeeze(np.zeros((1, 2, 1), dtype=np.float32), 0))
squeezed_by_keyword(np.squeeze(a=np.zeros((1, 2, 1), dtype=np.float32), axis=-1))
squeezed_list(np.squeeze([[1.5, 2.5]]))
variable_of_squeezed(tf.Variable(np.squeeze(np.zeros((1, 2, 4), dtype=np.float32))))

clipped(np.clip(boxes, 0, 5))
clipped_from_below(np.clip(boxes, 0.5, None))
clipped_by_keywords(np.clip(a=boxes, a_min=None, a_max=2.5))
clipped_by_negative(np.clip(boxes, -1, 1))
clipped_by_folded(np.clip(boxes, 0, 2 + 3))
clipped_complex(np.clip(spectrum, 0, 5))
clipped_doubles(np.clip(doubles, 0, 5))
clipped_doubles_by_floats(np.clip(doubles, 0.5, 2.5))
clipped_integers(np.clip(ids, 0, 3))
clipped_integers_by_floats(np.clip(ids, 0.0, 1.0))
clipped_bytes(np.clip(pixels, 0, 255))
clipped_bytes_by_negative(np.clip(pixels, -1, 1))
clipped_by_wide_integer(np.clip(boxes, 0, 70000))
clipped_by_wide_negative(np.clip(boxes, -70000, 0))
clipped_by_wide_float(np.clip(boxes, 0, 1e39))
clipped_bytes_by_wide(np.clip(pixels, 0, 300))
clipped_integers_by_wide(np.clip(ids, 0, 1099511627776))
clipped_integers_by_wide_negative(np.clip(ids, -1099511627776, 3))
clipped_by_array(np.clip(boxes, lows, 5))
clipped_by_upper_array(np.clip(boxes, 0, highs))
clipped_by_unread(np.clip(boxes, 0, time.time()))
clipped_by_booleans(np.clip(boxes, False, True))
for row in np.clip(boxes, 0, 5):
    clipped_row(row)
clipped_list(np.clip([1, 2, 3], 0, 2))
clipped_tuple(np.clip((1.5, 2.5, 3.5), 0, 2))
clipped_number(np.clip(5, 0, 3))
variable_of_clipped(tf.Variable(np.clip(boxes, 0, 5)))
