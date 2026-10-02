# https://github.com/wala/ML/issues/1009: `tf.pad` grows each extent by its `paddings` row and
# keeps the input's dtype.
import random

import tensorflow as tf


def consume_constant(x):
    return x


def consume_unresolved(x):
    return x


x = tf.ones([2, 3], dtype=tf.int32)
padded = tf.pad(x, [[1, 2], [0, 3]])
assert padded.shape == (5, 6) and padded.dtype == tf.int32
consume_constant(padded)

width = random.randint(1, 2)
unresolved = tf.pad(x, [[width, 0], [0, 0]])
assert unresolved.shape[1] == 3 and unresolved.dtype == tf.int32
consume_unresolved(unresolved)
