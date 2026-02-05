import tensorflow as tf


def test(t1, t2, t3):
    pass


# Positional (num_rows) and keyword (num_columns, dtype)
t1 = tf.eye(2, num_columns=3, dtype=tf.int32)
assert t1.shape == (2, 3)
assert t1.dtype == tf.int32

# Keyword (num_rows) and positional (none, since num_rows is first)
t2 = tf.eye(num_rows=3)
assert t2.shape == (3, 3)
assert t2.dtype == tf.float32  # default

# Mixed
t3 = tf.eye(2, dtype=tf.float32)
assert t3.shape == (2, 2)
assert t3.dtype == tf.float32

test(t1, t2, t3)
