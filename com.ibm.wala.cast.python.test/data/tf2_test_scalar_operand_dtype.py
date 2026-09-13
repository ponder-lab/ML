# Measurement fixture for wala/ML#922: the dtype of an elementwise op between a Python scalar
# literal and an array or tensor, by operand side and by what is known about the array.
import numpy as np
import tensorflow as tf


def consume_int_left_unknown_np(x):
    pass


def consume_int_right_unknown_np(x):
    pass


def consume_int_left_int64_np(x):
    pass


def consume_int_right_int64_np(x):
    pass


def consume_int_left_float32_tf(x):
    pass


def consume_int_right_float32_tf(x):
    pass


def consume_int_left_int32_tf(x):
    pass


def consume_float_left_int64_np(x):
    pass


def consume_float_left_float32_tf(x):
    pass


def bound():
    return int(np.random.randint(3, 9))


n = bound()
e = np.arange(
    0, n
)  # int64 array; the analysis cannot read its dtype (bounds are not constants)
assert e.dtype == np.int64
consume_int_left_unknown_np(2 * e)
assert (2 * e).dtype == np.int64
consume_int_right_unknown_np(e * 2)
assert (e * 2).dtype == np.int64

k = np.arange(0, 8)  # int64 array the analysis can read
assert k.dtype == np.int64
consume_int_left_int64_np(2 * k)
assert (2 * k).dtype == np.int64
consume_int_right_int64_np(k * 2)
assert (k * 2).dtype == np.int64

t = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # float32 tensor
consume_int_left_float32_tf(2 * t)
assert (2 * t).dtype == tf.float32
consume_int_right_float32_tf(t * 2)
assert (t * 2).dtype == tf.float32

i = tf.constant([[1, 2], [3, 4]])  # int32 tensor
consume_int_left_int32_tf(2 * i)
assert (2 * i).dtype == tf.int32

consume_float_left_int64_np(2.0 * k)
assert (2.0 * k).dtype == np.float64
consume_float_left_float32_tf(2.0 * t)
assert (2.0 * t).dtype == tf.float32
