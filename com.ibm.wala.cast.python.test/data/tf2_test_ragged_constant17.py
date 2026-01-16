# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


# A list of groups.
# Group 1 has one 2x3 matrix.
# Group 2 has two 2x3 matrices.
data = [
    [[[1, 1, 1], [2, 2, 2]]],  # Group 1
    [[[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]],  # Group 2
]

# We set ragged_rank=1.
# This means the OUTER list (the groups) varies in length.
# But everything INSIDE a group is a fixed uniform block.
t = tf.ragged.constant(data, None, 1)
assert t.shape == (2, None, 2, 3)
assert t.dtype == tf.int32

f(t)
