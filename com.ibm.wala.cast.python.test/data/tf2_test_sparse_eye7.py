import tensorflow as tf


def test(t1, t2):
    pass


# Positional (num_rows) and keyword (num_columns, dtype)
t1 = tf.sparse.eye(2, num_columns=3, dtype=tf.int32)
assert t1.shape == (2, 3)
assert t1.dtype == tf.int32

# Keyword args
t2 = tf.sparse.eye(num_rows=3)
assert t2.shape == (3, 3)
assert t2.dtype == tf.float32  # default

test(t1, t2)
