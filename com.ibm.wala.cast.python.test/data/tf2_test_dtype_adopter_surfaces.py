# Witnesses for the wala/ML#865 adopter surfaces: each allocator emits its API
# default only for a determinately absent dtype argument and degrades to unknown
# for a supplied one nothing can read. The unreadable spellings are runtime-valid
# but statically opaque (an unmodeled call result; NumPy's removed-in-1.24 alias,
# kept as the reported spelling as in tf2_test_dtype_absent_vs_unresolved.py).
import numpy as np
import tensorflow as tf


def ones_absent():
    return tf.ones([2, 2])


def ones_unreadable():
    return tf.ones([2, 2], dtype=tf.dtypes.as_dtype("float64"))


def zeros_absent():
    return tf.zeros([2, 2])


def zeros_unreadable():
    return tf.zeros([2, 2], dtype=tf.dtypes.as_dtype("float64"))


def eye_absent():
    return np.eye(3)


def eye_unreadable():
    return np.eye(3, dtype=np.int)


def range_absent():
    return tf.range(5)


def range_unreadable():
    return tf.range(5, dtype=tf.dtypes.as_dtype("float32"))


assert ones_absent().dtype == tf.float32
assert ones_unreadable().dtype == tf.float64
assert zeros_absent().dtype == tf.float32
assert zeros_unreadable().dtype == tf.float64
assert eye_absent().dtype == np.float64
assert eye_unreadable().dtype == np.int_
assert range_absent().dtype == tf.int32
assert range_unreadable().dtype == tf.float32
