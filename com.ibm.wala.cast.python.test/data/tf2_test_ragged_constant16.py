# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


data = [[[1, 2], [3, 4]], [[5, 6]]]

# Success: The data matches the inner shape (2,)
t1 = tf.ragged.constant(data, None, None, (2,))
assert t1.shape == (2, None, 2)
assert t1.dtype == tf.int32
# Output: (2, None, 2)

# Failure: You claim inner shape is (3,), but data is length 2
# t2 = tf.ragged.constant(data, ragged_rank=1, inner_shape=(3,))
# Raises ValueError: Inner shape mismatch

f(t1)
