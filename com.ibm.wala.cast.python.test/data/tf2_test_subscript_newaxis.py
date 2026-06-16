import tensorflow as tf


def consume(b):
    pass


a = tf.ones((3, 2), dtype=tf.float32)
# `a[:2, ..., tf.newaxis]`: slice the first axis to 2, ellipsis fills the remaining
# axis (2), and newaxis appends a size-1 axis -> (2, 2, 1).
b = a[:2, ..., tf.newaxis]
assert b.shape == (2, 2, 1) and b.dtype == tf.float32
consume(b)
