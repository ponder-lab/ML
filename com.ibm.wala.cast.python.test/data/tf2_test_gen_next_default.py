# Reproducer for wala/ML#699: the two-argument `next(it, default)` form dropped the default's flow.
# The generator is empty, so at runtime `next` returns the default; statically the default's type
# must reach the result. With the bug, the default contributed nothing and the type was dropped.
import tensorflow as tf


def consume(imgs):
    assert imgs.shape == (4, 4)
    assert imgs.dtype == tf.float32


def empty_batches():
    return
    yield  # unreachable; the `yield` keyword makes this a generator that yields nothing


batches = empty_batches()
imgs = next(batches, tf.ones((4, 4)))
consume(imgs)
