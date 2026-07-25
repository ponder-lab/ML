# The `tf.bool` dtype token (wala/ML#793): an explicit `dtype=tf.bool` on an allocator must
# resolve to the boolean dtype rather than falling to the float32 default.
import tensorflow as tf


def consume(m):
    pass


mask = tf.zeros((2, 2), dtype=tf.bool)
assert mask.dtype == tf.bool
assert mask.shape == (2, 2)
consume(mask)
