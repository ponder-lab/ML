# Test https://github.com/wala/ML/issues/978: a generator argument whose shape is unknown must
# degrade the result, not crash the analysis. `tf.where` over a runtime comparison has a
# data-dependent row count, which the analysis reads as an unknown shape.
import tensorflow as tf


def consume_row_starts(x):
    pass


def consume_row_limits(x):
    pass


def consume_row_splits(x):
    pass


def consume_row_lengths(x):
    pass


def consume_unknown_values(x):
    pass


def consume_unknown_values_rowids(x):
    pass


tokens = tf.constant([1, 2, 1, 3])
# Positions of the ones: data-dependent length, so its shape is unknown to the analysis.
starts = tf.squeeze(tf.where(tf.equal(tokens, 1)), -1)
assert starts.shape == (2,) and starts.dtype == tf.int64

r = tf.RaggedTensor.from_row_starts(tokens, starts)
assert r.shape.rank == 2
consume_row_starts(r)

limits = tf.concat([starts[1:], [4]], 0)
consume_row_limits(tf.RaggedTensor.from_row_limits(tokens, limits))

splits = tf.concat([starts, [4]], 0)
consume_row_splits(tf.RaggedTensor.from_row_splits(tokens, splits))

lengths = limits - starts
consume_row_lengths(tf.RaggedTensor.from_row_lengths(tokens, lengths))

# The values, not the partition, of unknown shape: their trailing axes, and so the result's rank,
# are unknown.
vals = tf.squeeze(tf.where(tf.equal(tokens, 1)), -1)
uv = tf.RaggedTensor.from_row_starts(vals, [0, 1])
assert uv.shape.rank == 2
consume_unknown_values(uv)

uvr = tf.RaggedTensor.from_value_rowids(vals, [0, 0])
assert uvr.shape.rank == 2
consume_unknown_values_rowids(uvr)
