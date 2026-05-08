import tensorflow as tf


def f(a):
    pass


# Multi-rank `tf.concat` with `axis=1` (positional non-default axis). Exercises the
# rank-aware branches in `Concat.computeConcatenatedShape`: axis normalization for
# multi-dim shapes, dim-preservation outside the concat axis, and the rank-equality
# guard.
t1 = tf.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])  # (2, 3)
result = tf.concat([t1, t2], axis=1)
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 6)
assert result.dtype == tf.int32
f(result)
