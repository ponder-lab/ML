# Test https://github.com/wala/ML/issues/997: `tf.keras.utils.unpack_x_y_sample_weight` outside
# `fit`, on a dataset element read in a loop, on a tuple literal, and on a bare tensor.
import numpy as np
import tensorflow as tf


def consume_loop_inputs(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    return x


def consume_loop_targets(y):
    assert y.shape == (2, 3) and y.dtype == tf.float32
    return y


def consume_literal_inputs(x):
    assert x.shape == (5, 1) and x.dtype == tf.float32
    return x


def consume_bare(x):
    assert x.shape == (6, 2) and x.dtype == tf.float32
    return x


dataset = tf.data.Dataset.from_tensor_slices(
    (np.ones((8, 4), dtype=np.float32), np.ones((8, 3), dtype=np.float32))
).batch(2)
for data in dataset:
    # A dataset element: the unpacked inputs and targets are its components.
    x, y, w = tf.keras.utils.unpack_x_y_sample_weight(data)
    consume_loop_inputs(x)
    consume_loop_targets(y)

# A tuple literal: its components.
a, b, c = tf.keras.utils.unpack_x_y_sample_weight((tf.ones((5, 1)), tf.ones((5, 3))))
consume_literal_inputs(a)

# A bare tensor: the data itself is the inputs.
t, u, v = tf.keras.utils.unpack_x_y_sample_weight(tf.ones((6, 2)))
consume_bare(t)
