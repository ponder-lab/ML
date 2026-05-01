# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


# TensorFlow sees an empty list and gives up on the inner dimensions.
t = tf.ragged.constant([], None, 1)
assert t.shape == (0, None)
assert t.dtype == tf.float32
# Output: (0, None) -> It lost the inner dimension info!

f(t)
