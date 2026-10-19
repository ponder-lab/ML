# Shadowed by the root's `pkg/sub/mod.py` for `pkg.sub.mod`, but run as a script of its own.
import tensorflow as tf


def take():
    return tf.ones((4, 4))


def consume_shadowed(x):
    assert x.shape == (7,)
    assert x.dtype == tf.float32


consume_shadowed(tf.ones((7,)))
