# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


# You tell TensorFlow: "Even though it's empty, if there WERE data,
# it would be shape (3,)."
t = tf.ragged.constant([], None, 1, (3,))
assert t.shape == (0, None, 3)
assert t.dtype == tf.float32
# Output: (0, None, 3) -> The inner structure is preserved.

f(t)
