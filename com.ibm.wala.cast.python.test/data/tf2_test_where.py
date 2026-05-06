import tensorflow as tf


def f(a):
    pass


# `tf.where(condition, x, y)` selects per-element from `x` or `y` based on
# `condition`. Output shape is the broadcast of all three (here all `(3,)`,
# so result is `(3,)`); output dtype matches `x` (and `y`, which TF
# requires to be the same dtype as `x`).
condition = tf.constant([True, False, True])
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([10.0, 20.0, 30.0])
result = tf.where(condition, x, y)
assert isinstance(result, tf.Tensor)
assert result.shape == (3,)
assert result.dtype == tf.float32
f(result)
