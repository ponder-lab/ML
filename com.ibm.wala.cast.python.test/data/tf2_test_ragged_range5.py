import tensorflow as tf


def f(x):
    pass


# Example 1: tf.ragged.range([3, 5, 2])
# Equivalent to:
# [0, 1, 2]
# [0, 1, 2, 3, 4]
# [0, 1]
r = tf.ragged.range([3, 5, 2])
assert isinstance(r, tf.RaggedTensor)
assert r.shape == (3, None)
assert r.dtype == tf.int32

f(r)
