import numpy as np
import tensorflow as tf


def consume_literal(strips):
    assert strips.shape == (3,) and strips.dtype == np.int64


def consume_concat_literal(edge_index):
    assert edge_index.shape == (3, 2) and edge_index.dtype == tf.int64


def consume_offsets(strips):
    assert strips.shape == (3,) and strips.dtype == np.int64


def consume(edge_index):
    assert edge_index.shape == (3, 2) and edge_index.dtype == tf.int64


# The per-graph edge lists of a batch, each already int64, and the node counts that offset them.
pieces = [np.array([[0, 1], [1, 2]]), np.array([[0, 1]])]

# The running sum of a literal list: numpy's platform integer.
literal = np.cumsum([0, 3, 2])
consume_literal(literal)
consume_concat_literal(tf.concat([w + literal[i] for i, w in enumerate(pieces)], 0))

# The batch merger's own form: a literal prefix concatenated with a computed list.
sizes = [3, 2]
strips = np.cumsum([0] + sizes)
consume_offsets(strips)
consume(tf.concat([w + strips[i] for i, w in enumerate(pieces)], 0))


def consume_axis(c):
    assert c.shape == (2, 2) and c.dtype == np.int64


def consume_dtype(c):
    assert c.shape == (3,) and c.dtype == np.float64


consume_axis(np.cumsum(pieces[0], axis=0))
consume_dtype(np.cumsum([0, 3, 2], dtype=np.float64))


def consume_widened(c):
    assert c.shape == (2,) and c.dtype == np.int64


consume_widened(np.cumsum(np.array([1, 2], dtype=np.int32)))


def consume_uint8(c):
    # numpy sums an unsigned narrow input as uint64, which the analysis has no dtype for.
    assert c.shape == (2,) and c.dtype == np.uint64


def consume_computed_axis(c):
    assert c.shape == (2, 2) and c.dtype == np.int64


consume_uint8(np.cumsum(np.array([1, 2], dtype=np.uint8)))
last = len(pieces[0].shape) - 1
consume_computed_axis(np.cumsum(pieces[0], axis=last))
