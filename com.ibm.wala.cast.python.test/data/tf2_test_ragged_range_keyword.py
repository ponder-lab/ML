import tensorflow as tf


def test(x1, x2, x3):
    pass


# Positional
x1 = tf.ragged.range(5)
# Shape is [1, None] because inputs are scalars.
# The internal values shape is [5].
assert x1.shape.as_list() == [1, None]
assert x1.dtype == tf.int32

# Keyword 'starts' (acting as limit because single argument)
x2 = tf.ragged.range(starts=5)
assert x2.shape.as_list() == [1, None]
assert x2.dtype == tf.int32

# Keyword 'starts', 'limits', 'deltas'
x3 = tf.ragged.range(starts=0, limits=5, deltas=1)
assert x3.shape.as_list() == [1, None]
assert x3.dtype == tf.int32

test(x1, x2, x3)
