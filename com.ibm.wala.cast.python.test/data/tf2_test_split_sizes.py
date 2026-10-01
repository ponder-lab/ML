# Test https://github.com/wala/ML/issues/993: `tf.split` by a list of sizes gives pieces whose
# extent along the split axis is one of the listed sizes.
import tensorflow as tf


def consume_equal(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def consume_unequal(x):
    assert x.shape in ((1, 3), (3, 3)) and x.dtype == tf.float32


def consume_inferred(x):
    assert x.shape in ((1, 3), (3, 3)) and x.dtype == tf.float32


def consume_axis(x):
    assert x.shape in ((4, 1), (4, 2)) and x.dtype == tf.float32


t = tf.ones((4, 3))

for piece in tf.split(t, [2, 2], 0):
    consume_equal(piece)

for piece in tf.split(t, [1, 3], 0):
    consume_unequal(piece)

for piece in tf.split(t, [1, -1], 0):
    consume_inferred(piece)

for piece in tf.split(t, [1, 2], 1):
    consume_axis(piece)
