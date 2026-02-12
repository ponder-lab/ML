# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/reduce_mean#for_example
import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def h(a):
    pass


x = tf.constant([[1.0, 1.0], [2.0, 2.0]])

r1 = tf.reduce_mean(x)
assert r1.shape == ()
assert r1.dtype == tf.float32
f(r1)

r2 = tf.reduce_mean(x, 0)
assert r2.shape == (2,)
assert r2.dtype == tf.float32
g(r2)

r3 = tf.reduce_mean(x, 1)
assert r3.shape == (2,)
assert r3.dtype == tf.float32
h(r3)
