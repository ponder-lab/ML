import tensorflow as tf


def test_none_dtype(t):
    pass


# passing None as dtype should use default (float32 for float inputs)
t = tf.constant([1.0, 2.0], dtype=None)
assert t.shape.as_list() == [2]
assert t.dtype == tf.float32

test_none_dtype(t)
