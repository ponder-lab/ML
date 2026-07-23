# Probe for wala/ML#743: the unpassed integer default `n` determines the constructed
# tensor's leading dimension, so the shape resolves only if the default materializes in
# the pointer analysis.
import tensorflow as tf


def consume(t):
    pass


def make(n=4):
    return tf.ones((n, 2))


y = make()
consume(y)

assert y.shape == (4, 2)
assert y.dtype == tf.float32
