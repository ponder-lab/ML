# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


# Two groups. Group 1 has one matrix. Group 2 has two matrices.
data = [[[[1, 1], [2, 2]]], [[[3, 3], [4, 4]], [[5, 5], [6, 6]]]]  # Group 1  # Group 2

# We set ragged_rank=1 (The groups are ragged).
# We set inner_shape=(2, 2) (The things inside are 2x2 matrices).
t = tf.ragged.constant(data, None, 1, (2, 2))
assert t.shape == (2, None, 2, 2)
assert t.dtype == tf.int32

f(t)
