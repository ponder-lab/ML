# An integer subscript of a tensor, of a tensor's dynamic shape, and of a tensor merged with a
# Python scalar (wala/ML#1225).
import tensorflow as tf


def consume_tensor_index(n):
    assert n.shape == (5,) and n.dtype == tf.float32


def consume_shape_index(n):
    assert n.shape == () and n.dtype == tf.int32


def consume_phi(n):
    assert n.shape == () and n.dtype == tf.int32


def consume_row_of_index(n):
    assert n.shape == (4,) and n.dtype == tf.int32


c = tf.ones([3, 5])
consume_tensor_index(c[0])
consume_shape_index(tf.shape(c)[0])
consume_phi(2 if len(c) > 5 else tf.shape(c)[0])
edge_index = tf.constant([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=tf.int32)
row = edge_index[0]
consume_row_of_index(row)
