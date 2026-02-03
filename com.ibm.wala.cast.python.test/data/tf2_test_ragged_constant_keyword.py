import tensorflow as tf


def test(x1, x2):
    pass


# Positional
x1 = tf.ragged.constant([[1, 2], [3]])
assert x1.shape.as_list() == [2, None]
assert x1.dtype == tf.int32

# Keyword
x2 = tf.ragged.constant(pylist=[[1, 2], [3]])
assert x2.shape.as_list() == [2, None]
assert x2.dtype == tf.int32

test(x1, x2)
