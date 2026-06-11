import tensorflow as tf


def f(a):
    pass


# Counterpart of `tf2_test_sequence_mask.py` that passes the optional `dtype`
# override (`tf.sequence_mask(..., dtype=tf.int32)`). The output dtype follows
# the argument (int32 here) instead of the default `bool`, and the constant
# `maxlen` gives the precise (3, 5) shape.
y = tf.sequence_mask([1, 3, 2], maxlen=5, dtype=tf.int32)
assert isinstance(y, tf.Tensor)
assert y.shape == (3, 5)
assert y.dtype == tf.int32

f(y)
