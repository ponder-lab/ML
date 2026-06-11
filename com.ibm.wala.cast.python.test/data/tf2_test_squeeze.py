import tensorflow as tf


def consume_axis(a):
    pass


def consume_all(b):
    pass


x = tf.ones((2, 1, 3, 1), dtype=tf.float32)
assert x.shape == (2, 1, 3, 1) and x.dtype == tf.float32

# Squeeze a named axis (axis 1, size 1): drops just that axis -> (2, 3, 1).
a = tf.squeeze(x, [1])
assert a.shape == (2, 3, 1)
consume_axis(a)

# Squeeze with no axis: drops every size-1 axis -> (2, 3).
b = tf.squeeze(x)
assert b.shape == (2, 3)
consume_all(b)
