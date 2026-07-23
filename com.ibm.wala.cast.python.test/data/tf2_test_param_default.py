# Probe for wala/ML#743: an unpassed parameter default (the float scalar `eps`) must
# materialize in the pointer analysis for the elementwise addition to resolve. The result
# keeps the tensor operand's shape (scalar broadcast) and dtype.
import tensorflow as tf


def consume(t):
    pass


def add_eps(x, eps=1e-6):
    return x + eps


y = add_eps(tf.ones((2, 3)))
consume(y)

assert y.shape == (2, 3)
assert y.dtype == tf.float32
