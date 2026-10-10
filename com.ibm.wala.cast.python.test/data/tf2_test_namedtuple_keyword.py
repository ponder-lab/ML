# A `collections.namedtuple` built by keyword: reading a field back gives the value passed for it,
# as reading a positionally built one does.
import collections

import tensorflow as tf

Hypothesis = collections.namedtuple("Hypothesis", ("index", "states"))


def consume_keyword(x):
    assert x.dtype == tf.int32
    assert x.shape == ()


def consume_positional(x):
    assert x.dtype == tf.int32
    assert x.shape == ()


def consume_loop(x):
    assert x.dtype == tf.int32
    assert x.shape == (1, 1)


keyword = Hypothesis(index=tf.constant(3), states=tf.zeros((2,)))
consume_keyword(keyword.index)
positional = Hypothesis(tf.constant(4), tf.zeros((2,)))
consume_positional(positional.index)


def body(t, h):
    consume_loop(tf.reshape(h.index, [1, 1]))
    return t + 1, Hypothesis(index=h.index, states=h.states)


tf.while_loop(lambda t, h: t < 2, body, [tf.constant(0), keyword])
