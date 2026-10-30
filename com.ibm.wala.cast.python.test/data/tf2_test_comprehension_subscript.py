# Test a subscript of a tensor or ndarray a list comprehension captures from its enclosing function,
# indexed by the comprehension variable or by a constant.
import numpy as np
import tensorflow as tf


def consume_loop(row):
    assert row.shape == (2,) and row.dtype == np.int32


def consume_comprehension(row):
    assert row.shape == (2,) and row.dtype == np.int32


def consume_tensor_row(row):
    assert isinstance(row, tf.Tensor)
    assert row.shape == (2,) and row.dtype == tf.float32


def consume_constant_row(row):
    assert isinstance(row, tf.Tensor)
    assert row.shape == (2,) and row.dtype == tf.float32


def consume_parameter_index(row):
    pass


def pick(t, i):
    # An int parameter outside a comprehension is not read as an index.
    consume_parameter_index(t[i])


def calc_shapes(meta):
    return tf.cast(meta[..., 3:5], tf.int32).numpy()


def rows():
    meta = tf.ones([3, 6])
    shapes = calc_shapes(meta)

    for i in range(shapes.shape[0]):
        consume_loop(shapes[i])

    [consume_comprehension(shapes[i]) for i in range(shapes.shape[0])]

    t = tf.ones([3, 2])
    [consume_tensor_row(t[i]) for i in range(3)]
    [consume_constant_row(t[0]) for _ in range(3)]
    pick(t, 1)


rows()
