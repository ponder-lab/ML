# Test: two functions whose results flow into each other, so the generator dispatch of either
# result reaches the other's and comes back to its own.
import tensorflow as tf


def even(x, n):
    if n == 0:
        return x
    return odd(x, n - 1)


def odd(x, n):
    if n == 0:
        return tf.ones([2])
    return even(x, n - 1)


def consume(t):
    return t


r = even(tf.zeros([2]), 3)
assert r.shape == (2,), r.shape
consume(r)
s = odd(tf.zeros([2]), 3)
assert s.shape == (2,), s.shape
consume(s)
