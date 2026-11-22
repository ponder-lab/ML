# A helper that converts its argument only when the argument has some other type, so the type test
# guarding each conversion decides which arm runs (wala/ML#1053).
import numpy as np
import tensorflow as tf


def consume_float32_kept(x):
    pass


def consume_float64_converted(x):
    pass


def consume_float32_unequal(x):
    pass


def consume_tensor_kept(x):
    pass


def consume_tensor_mixed_token(x):
    pass


def consume_array_kept(x):
    pass


def consume_list_converted(x):
    pass


def consume_cast_x_kept(x):
    pass


def consume_cast_x_converted(x):
    pass


def consume_identity_kept(x):
    pass


def consume_cast_x_chained(x):
    pass


def to_float32(x):
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    return x


def to_int32(x):
    # A conversion to another dtype than the argument's, so its arm's value is seen if it is kept.
    if x.dtype == np.float64:
        x = x.astype(np.int32)
    return x


def float32_kept():
    x = to_int32(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_float32_kept(x)


def float64_converted():
    x = to_float32(np.zeros((2, 3)))
    assert x.dtype == np.float32
    consume_float64_converted(x)


def to_float32_unequal(x):
    if x.dtype != np.float64:
        return x
    return x.astype(np.int32)


def float32_unequal():
    x = to_float32_unequal(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_float32_unequal(x)


def cast_tensor(x):
    if x.dtype == tf.float64:
        x = tf.cast(x, tf.float32)
    return x


def cast_tensor_int32(x):
    if x.dtype == tf.float64:
        x = tf.cast(x, tf.int32)
    return x


def tensor_kept():
    x = cast_tensor_int32(tf.ones([2, 3]))
    assert x.dtype == tf.float32
    consume_tensor_kept(x)


def tensor_mixed_token():
    # A NumPy array's dtype equals the TensorFlow token of the same type.
    x = cast_tensor(np.zeros((2, 3)))
    assert x.dtype == tf.float32
    consume_tensor_mixed_token(x)


def from_list(x):
    if isinstance(x, list):
        x = np.array(x, dtype=np.int32)
    return x


def array_kept():
    x = from_list(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_array_kept(x)


def list_converted():
    x = from_list([1, 2, 3])
    assert x.dtype == np.int32
    consume_list_converted(x)


def cast_x(x):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray) and x.dtype == np.float64:
        x = x.astype(np.float32)
    return x


def cast_x_kept():
    x = cast_x(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_cast_x_kept(x)


def cast_x_converted():
    x = cast_x(np.zeros((2, 3)))
    assert x.dtype == np.float32
    consume_cast_x_converted(x)


def cast_x_int(x):
    # The issue's helper with conversions to other dtypes than the argument's, so a kept arm's value
    # is seen: the second guard reads `x` after the first, through the first's merge.
    if isinstance(x, list):
        x = np.array(x, dtype=np.int64)
    if isinstance(x, np.ndarray) and x.dtype == np.float64:
        x = x.astype(np.int32)
    return x


def cast_x_chained():
    x = cast_x_int(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_cast_x_chained(x)


def to_float32_identity(x):
    # `is` compares a dtype with the type object, which is never the same object.
    if x.dtype is np.float64:
        x = x.astype(np.float32)
    return x


def identity_kept():
    x = to_float32_identity(np.zeros((2, 3)))
    assert x.dtype == np.float64
    consume_identity_kept(x)


float32_kept()
float64_converted()
float32_unequal()
tensor_kept()
tensor_mixed_token()
array_kept()
list_converted()
cast_x_kept()
cast_x_converted()
cast_x_chained()
identity_kept()
