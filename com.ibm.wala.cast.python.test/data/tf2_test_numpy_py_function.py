# Test the dtype of `tf.numpy_function`'s and `tf.py_function`'s result: the dtype `Tout` states, or
# each of a list's.
import numpy as np
import tensorflow as tf


def single_sink(x):
    return x


def first_sink(x):
    return x


def second_sink(x):
    return x


def py_sink(x):
    return x


def double(x):
    return x * 2


def split(x):
    return x.astype(np.float32), x.astype(np.int32)


x = tf.constant([1.0, 2.0])
single = tf.numpy_function(double, [x], tf.float32)
single_sink(single)
assert single.dtype == tf.float32
first, second = tf.numpy_function(split, [x], [tf.float32, tf.int32])
first_sink(first)
second_sink(second)
assert first.dtype == tf.float32 and second.dtype == tf.int32
py = tf.py_function(double, [x], tf.float32)
py_sink(py)
assert py.dtype == tf.float32
