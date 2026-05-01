import tensorflow as tf


def test(x1, x2, x3):
    pass


# Positional
x1 = tf.ragged.constant([[1, 2], [3]])
assert x1.shape.as_list() == [2, None]
assert x1.dtype == tf.int32

# Keyword
x2 = tf.ragged.constant(pylist=[[1, 2], [3]])
assert x2.shape.as_list() == [2, None]
assert x2.dtype == tf.int32

# Mixed
x3 = tf.ragged.constant([[1, 2], [3]], ragged_rank=1)
assert x3.shape.as_list() == [2, None]
assert x3.dtype == tf.int32

test(x1, x2, x3)
