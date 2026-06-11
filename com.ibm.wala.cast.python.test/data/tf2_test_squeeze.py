import tensorflow as tf


def consume_axis(a):
    pass


def consume_all(b):
    pass


def consume_single(c):
    pass


def consume_multi(d):
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

# A single (non-list) integer axis -> (2, 3, 1).
c = tf.squeeze(x, 1)
assert c.shape == (2, 3, 1)
consume_single(c)

# Multiple named axes -> (2, 3).
d = tf.squeeze(x, [1, 3])
assert d.shape == (2, 3)
consume_multi(d)
