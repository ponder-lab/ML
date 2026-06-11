import tensorflow as tf


def consume_rows(r):
    pass


def consume_cols(c):
    pass


def consume_index(i):
    pass


x = tf.ones((4, 5, 6), dtype=tf.float32)
assert x.shape == (4, 5, 6) and x.dtype == tf.float32

# Non-zero start on the first axis: `[1:3]` keeps 2 rows, trailing axes intact.
rows = x[1:3, :, :]
assert rows.shape == (2, 5, 6)
consume_rows(rows)

# Slice on the middle axis: `[:, 1:, :]` drops its leading element (5 -> 4).
cols = x[:, 1:, :]
assert cols.shape == (4, 4, 6)
consume_cols(cols)

# Integer index on the middle axis drops that axis entirely: `[:, 0, :]` -> (4, 6).
idx = x[:, 0, :]
assert idx.shape == (4, 6)
consume_index(idx)
