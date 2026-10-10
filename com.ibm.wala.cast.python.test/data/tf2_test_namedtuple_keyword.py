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


Point = collections.namedtuple("Point", "x y")


def consume_string_x(x):
    assert x.dtype == tf.float32
    assert x.shape == ()


def consume_indexed(x):
    assert x.dtype == tf.int32
    assert x.shape == ()


def consume_unpacked(x):
    assert x.dtype == tf.float32
    assert x.shape == ()


def consume_starred(x):
    assert x.dtype == tf.float32
    assert x.shape == (3,)


point = Point(tf.constant(1.0), y=tf.constant(2))
consume_string_x(point.x)
consume_indexed(point[1])
first, _ = point
consume_unpacked(first)
spread = Point(tf.constant(1.0), *[tf.zeros((3,))])
consume_starred(tf.convert_to_tensor(spread.y))


def consume_listed(x):
    assert x.dtype == tf.float32
    assert x.shape == (2,)


listed = Point([tf.ones((2,))], y=0)
consume_listed(listed.x[0])
