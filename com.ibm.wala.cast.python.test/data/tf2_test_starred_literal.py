# Test https://github.com/wala/ML/issues/989: a starred element in a tuple or list literal unpacks
# its iterable's elements into the literal rather than standing for one element.
import numpy as np
import tensorflow as tf


def consume_index(x):
    assert x.shape == (3,)


def consume_element(x):
    assert x.dtype in (tf.float32, tf.int32)


def consume_reshaped(x):
    assert x.shape == (4, 3)


def reshape_starred(t, rest):
    consume_reshaped(tf.reshape(t, (*rest, 3)))


def consume_inline(x):
    assert x.dtype in (tf.float32, tf.int32)


def consume_array_star(x):
    pass


def iterate_inline(a, b, g):
    for x in [a, *[b, g]]:
        consume_inline(x)


def iterate_array_star(a):
    for x in [a, *np.ones((2, 4))]:
        consume_array_star(x)


def index_after_star(head):
    leading = (5, *head)
    consume_index(np.zeros(leading[1]))


def iterate(rest, a):
    for x in [a, *rest]:
        consume_element(x)


index_after_star([3, 4])
iterate([tf.ones((2,)), tf.zeros((3,), dtype=tf.int32)], tf.ones((4,)))
reshape_starred(tf.ones((12,)), [4])
iterate_inline(tf.ones((4,)), tf.ones((2,)), tf.zeros((3,), dtype=tf.int32))
iterate_array_star(tf.ones((4,)))
