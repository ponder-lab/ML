import tensorflow as tf


def f(a):
    pass


# `tf.concat` with negative axis (`axis=-1`). Exercises the `axis < 0 ?
# axis + rank : axis` normalization branch in `Concat.computeConcatenatedShape`.
# For 2D inputs, `axis=-1` is the last axis (index 1).
t1 = tf.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])  # (2, 3)
result = tf.concat([t1, t2], axis=-1)
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 6)
assert result.dtype == tf.int32
f(result)
