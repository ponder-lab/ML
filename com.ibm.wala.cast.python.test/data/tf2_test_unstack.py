import tensorflow as tf


def consume(a, b):
    assert a.shape == (3, 4) and a.dtype == tf.float32
    assert b.shape == (3, 4) and b.dtype == tf.float32


def consume_axis(a, b, c):
    assert a.shape == (2, 4) and a.dtype == tf.float32
    assert c.shape == (2, 4) and c.dtype == tf.float32


x = tf.ones((2, 3, 4))
first, second = tf.unstack(x)
consume(first, second)

p, q, r = tf.unstack(x, axis=1)
consume_axis(p, q, r)
