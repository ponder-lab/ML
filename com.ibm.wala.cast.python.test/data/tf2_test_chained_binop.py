import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0, 6.0])
z = tf.constant([7.0, 8.0, 9.0])

inner = x + y
assert inner.shape == (3,)
assert inner.dtype == tf.float32

outer = inner * z
assert outer.shape == (3,)
assert outer.dtype == tf.float32

f(outer)
