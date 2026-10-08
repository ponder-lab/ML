# Test `a or b` and `a and b`, which evaluate to one of their operands, never to a bool they make up:
# `x.shape[2] or tf.shape(x)[2]` is the static extent when it is known, as a beam-search helper reads
# a beam width, `k or 7` is `k` itself when `k` is truthy, and `k and 5` is `k` itself when `k` is
# falsy.
import tensorflow as tf


def consume_or(x):
    assert x.shape == (3,) and x.dtype == tf.int32
    return x


def consume_beam(x):
    assert x.shape == (3,) and x.dtype == tf.int32
    return x


def consume_and(x):
    assert x.shape in [(5,), (0,)] and x.dtype == tf.int32
    return x


def beam(parent_ids):
    beam_width = parent_ids.shape[2] or tf.shape(parent_ids)[2]
    consume_beam(tf.range(beam_width))


def make_or(k):
    n = k or 7
    consume_or(tf.range(n))


def make_and(k):
    n = k and 5
    consume_and(tf.range(n))


beam(tf.ones((5, 2, 3), dtype=tf.int32))
make_or(3)
make_and(2)
make_and(0)
