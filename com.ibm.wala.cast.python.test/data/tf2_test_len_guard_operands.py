# Test the `len(...)` rank-guard fold (wala/ML#1020) on operands that are not shape vectors: `len(t)`
# of a tensor is its first extent, and `len(xs)` of a list is the list's length. Neither is a rank, so
# a fold that read either as one would decide these guards wrong and drop the arm that runs.
import tensorflow as tf


def consume_a(t):
    assert t.shape == (3, 5)
    return t


def consume_b(t):
    assert t.shape == (3, 5)
    return t


def f(t, xs):
    if len(t) == 3:
        consume_a(t)
    if len(xs) == 3:
        consume_b(xs[0])


t = tf.ones((3, 5))
f(t, [t, t, t])
