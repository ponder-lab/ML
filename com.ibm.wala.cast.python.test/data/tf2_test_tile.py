import tensorflow as tf


def consume_tile(a):
    pass


# tf.tile multiplies each axis by the corresponding multiple: (2, 3) tiled by
# [2, 1] -> (4, 3).
t = tf.tile(tf.ones((2, 3)), [2, 1])
assert isinstance(t, tf.Tensor)
assert t.shape == (4, 3)
assert t.dtype == tf.float32
consume_tile(t)
